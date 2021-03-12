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
TODO: As above, focussing especially on the IR bands.
"""
import os
import numpy as np
import wandb
import xgboost as xgb
from sklearn import metrics
import xarray as xr
import dask.array as da
import dask.distributed
from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, SAT_DIR
from src.preprocessing.esa_compress import compress_esa, decompress_esa, FORW_D, REV_D
from src.preprocessing.load_landsat_esa import (
    return_xy_npa,
    y_npa_to_xr,
    x_npa_to_xr,
    return_x_y_da,
)
from src.utils import timeit
from src.preprocessing.landsat_to_ncf import create_netcdfs
from src.visualisation.ani import animate_prediction


@timeit
def train_xgb(
    train_X,
    train_Y,
    test_X,
    test_Y,
    x_da,
    y_da,
    cfd,
    objective="multi:softmax",
    eta=0.3,
    max_depth=12,
    nthread=16,
    num_round=20,
    use_dask=False,
):
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
    # setup parameters for xgboost
    param = {}
    param["objective"] = objective  # use softmax multi-class classification
    param["eta"] = eta  # scale weight of positive examples
    param["max_depth"] = max_depth  # max_depth
    param["silent"] = 1
    param["nthread"] = nthread  # number of threads
    param["num_class"] = np.max(train_Y) + 1  # max size of labels.
    wandb.config.update(param)

    if use_dask:
        # https://docs.dask.org/en/latest/array-api.html#dask.array.from_npy_stack
        # https://xgboost.readthedocs.io/en/latest/tutorials/dask.html
        dask_direc = os.path.join(SAT_DIR, "dask_temp")

        if not os.path.exists(dask_direc):
            print("making ", dask_direc)
            os.mkdir(dask_direc)

        cluster = dask.distributed.LocalCluster(n_workers=4, threads_per_worker=10)
        client = dask.distributed.Client(cluster)

        def daskify(x):
            future = client.scatter(x)
            x = da.from_delayed(future, shape=x.shape, dtype=x.dtype)
            x = x.rechunk()
            return x.persist()

        # train_X,
        # train_Y,
        # test_X,
        # test_Y,

        dtrain = xgb.dask.DaskDMatrix(client, daskify(train_X), daskify(train_Y))
        dtest = xgb.dask.DaskDMatrix(client, daskify(test_X), daskify(test_Y))

        del train_X
        del train_Y
        del test_X
        del test_Y

        output = xgb.dask.train(
            client,
            param,
            dtrain,
            num_boost_round=num_round,
            evals=[(dtrain, "train"), (dtrain, "test")],
        )

        # x_all, y_all = return_xy_npa(
        #     x_da, y_da, year=range(cfd["start_year_i"], cfd["end_year_i"])
        # )  # all data as numpy.

        """
        xg_all = xgb.DMatrix(
            x_all, label=compress_esa(y_all)
        )  # pass all data to xgb data matrix
        y_pr_all = decompress_esa(
            bst.predict(xg_all)
        )  # predict whole time period using model
        x = dask.delayed(load_array_from_file)(fn)
        x = da.from_delayed(x, shape=(1000, 1000, 1000), dtype=float)
        x = x.rechunk((100, 100, 100))
        x = x.persist()
        """
        # prediction = xgb.dask.predict(client, output, dtrain)
        # Or equivalently, pass ``output['booster']``:
        # prediction = xgb.dask.predict(client, output['booster'], dtrain)
        prediction = xgb.dask.predict(client, output, dtest)
        print(prediction)
        # https://stackoverflow.com/questions/45941528/how-to-efficiently-send-a-large-numpy-array-to-the-cluster-with-dask-array
    else:
        # label need to be 0 to num_class -1
        xg_train = xgb.DMatrix(train_X, label=train_Y)  # make train DMatrix
        xg_test = xgb.DMatrix(test_X, label=test_Y)  # make test DMatrix
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
        bst.save_model(
            os.path.join(wandb.run.dir, wandb.run.name + "_xgb.model")
        )  # save model.
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
        print(
            "Classification accuracy: {}".format(
                metrics.accuracy_score(y_all, y_pr_all)
            )
        )
        # return bst


if __name__ == "__main__":
    # usage:  python3 src/models/xgb.py > log.txt
    # create_netcdfs() # uncomment to preprocess data.

    cfd = {
        "start_year_i": 8,  # python3 src/models/xgb.py
        "mid_year_i": 19,
        "end_year_i": 24,
        "take_esa_coords": True,  # True,  # False,
        "use_ffil": False,
        "use_mfd": False,
        "use_ir": False,
        "objective": "multi:softmax",
        "eta": 0.3,
        "max_depth": 12,
        "nthread": 16,
        "num_round": 20,
        "use_dask": False,
        "prefer_remake": True,
    }
    print("cfd:\n", cfd)

    wandb.init(project="xgbc-esa-cci", entity="sdat2")  # my id for wandb
    wandb.config.update(cfd)

    x_da, y_da = return_x_y_da(
        take_esa_coords=cfd["take_esa_coords"],
        use_ffil=cfd["use_ffil"],
        use_mfd=cfd["use_mfd"],
        use_ir=cfd["use_ir"],
        prefer_remake=cfd["prefer_remake"],
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
        x_tr,
        compress_esa(y_tr),
        x_te,
        compress_esa(y_te),
        x_da,
        y_da,
        cfd,
        num_round=cfd["num_round"],
        objective=cfd["objective"],
        eta=cfd["eta"],
        max_depth=cfd["max_depth"],
        nthread=cfd["nthread"],
        use_dask=cfd["use_dask"],
    )  # train xgboost model.
    wandb.log(cfd)
