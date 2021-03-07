"""
xgb.py
======

This module trains an xgboost classification model using preprocessed inputs.

X: Landsat data.
Y: ESA CCI.

Also animates the result.

Usage:

    python3 src/models/xgb.py

TODO: Should add hydra to allow a lot of different model hyperparameters to be passed in.
TODO: Implement xgboost dask to let the model run with full resolution data.
TODO: Implement GPU usage in xgboost.
TODO: Fix bug with loading full res data xarray.
TODO: Look at trends in landsat bands to see if preprocessing can be improved.

"""
import os
import numpy as np
import wandb
import xgboost as xgb
from sklearn import metrics
import xarray as xr
from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, SAT_DIR
from src.preprocessing.esa_compress import compress_esa, decompress_esa, FORW_D, REV_D
from src.preprocessing.load_landsat_esa import return_xy_npa, y_npa_to_xr, x_npa_to_xr, return_x_y_da
from src.utils import timeit
from src.preprocessing.landsat_to_ncf import create_netcdfs
from src.visualisation.ani import animate_prediction


@timeit
def train_xgb(train_X, train_Y, test_X, test_Y, 
              objective="multi:softmax", eta=0.3, max_depth=12,
              nthread=16, num_round=20):
    """
    Train an xgboost model using numpy inputs.
    :param train_X: npa, float32
    :param train_Y: npa, int16
    :param test_X: npa, float32
    :param test_Y: npa, int16

    significant algorithm parameter:

    eta [default=0.3, alias: learning_rate]

    Step size shrinkage used in update to prevents overfitting.
    After each boosting step, we can directly get the weights of new features,
    and eta shrinks the feature weights to make the boosting process more conservative.
    """
    # replace with your id.
    # label need to be 0 to num_class -1
    xg_train = xgb.DMatrix(train_X, label=train_Y)  # make train DMatrix
    xg_test = xgb.DMatrix(test_X, label=test_Y)  # make test DMatrix
    # setup parameters for xgboost
    param = {}
    param["objective"] = objective  # use softmax multi-class classification
    param["eta"] = eta # scale weight of positive examples
    param["max_depth"] = max_depth  # max_depth
    param["silent"] = 1
    param["nthread"] = nthread  # number of threads
    param["num_class"] = np.max(train_Y) + 1  # max size of labels.
    wandb.config.update(param)
    watchlist = [(xg_train, "train"), (xg_test, "test")]
    bst = xgb.train(
        param,
        xg_train,
        num_round,
        watchlist,
        callbacks=[wandb.xgboost.wandb_callback()],
    )
    # get prediction
    pred = bst.predict(xg_test)  # predict for the test set.
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print("Test error using softmax = {}".format(error_rate))
    wandb.log({"Error Rate": error_rate})
    return bst


if __name__ == "__main__":
    # usage:  python3 src/models/xgb.py > log.txt
    # create_netcdfs() # uncomment to preprocess data.
    cfd = {
        "start_year_i": 8,
        "mid_year_i": 19,
        "end_year_i": 24,
        "take_esa_coords": True, # False,
        "use_ffil": True,
        "use_mfd": False,
        "use_ir": False,
        "objective": "multi:softmax",
        "eta": 0.3,
        "max_depth": 12,
        "nthread": 16,
        "num_round": 25
    }
    print("cfd:\n", cfd)
    wandb.init(project="xgbc-esa-cci", entity="sdat2")  # my id for wandb
    wandb.config.update(cfd)

    x_da, y_da = return_x_y_da(
        take_esa_coords=cfd["take_esa_coords"],
        use_ffil=cfd["use_ffil"],
        use_mfd=cfd["use_mfd"],
        use_ir=cfd["use_ir"],
    )  # load preprocessed data from netcdfs
    # there are now 24 years to choose from.
    # train set goes from 0 to 1. # print(x_da.year.values)
    # test_inversibility()
    # print("x_da", x_da)
    # print("y_da", y_da)

    x_tr, y_tr = return_xy_npa(
        x_da, y_da, year=range(cfd["start_year_i"], cfd["mid_year_i"])
    )  # load numpy train data.
    x_te, y_te = return_xy_npa(
        x_da, y_da, year=range(cfd["mid_year_i"], cfd["end_year_i"])
    )  # load numpy test data.
    bst = train_xgb(
        x_tr, compress_esa(y_tr), x_te, compress_esa(y_te)
    )  # train xgboost model.
    bst.save_model(
        os.path.join(wandb.run.dir, wandb.run.name + "_xgb.model")
    )  # save model.
    wandb.log(cfd)
    x_all, y_all = return_xy_npa(
        x_da, y_da, year=range(cfd["start_year_i"], cfd["end_year_i"])
    )  # all data as numpy.
    xg_all = xgb.DMatrix(
        x_all, label=compress_esa(y_all)
    )  # pass all data to xgb data matrix
    y_pr_all = decompress_esa(
        bst.predict(xg_all)
    )  # predict whole time period using model
    y_pr_da = y_npa_to_xr(
        y_pr_all, y_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"]))
    )  # transform full prediction to dataarray.
    y_pr_da.to_netcdf(
        os.path.join(wandb.run.dir, wandb.run.name + "_y.nc")
    )  # save to netcdf
    animate_prediction(
        x_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"])),
        y_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"])),
        y_pr_da,
        video_path=os.path.join(wandb.run.dir, wandb.run.name + "_joint_val.mp4"),
    )  # animate prediction vs inputs.
    print("Classification accuracy: {}".format(metrics.accuracy_score(y_all, y_pr_all)))
