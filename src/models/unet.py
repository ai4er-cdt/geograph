"""
xgb.py
======

This module trains an xgboost classification model using preprocessed inputs.

X: Landsat data.
Y: ESA CCI.

Also animates the result.

Usage:

    python3 src/models/unet.py

"""
import os
import numpy as np
import wandb
import xgboost as xgb
from sklearn import metrics
import xarray as xr

from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, SAT_DIR
from src.preprocessing.esa_compress import compress_esa, decompress_esa, FORW_D, REV_D
from src.preprocessing.load_landsat_esa import return_xy_np_grid, y_npa_to_xr, x_npa_to_xr, return_x_y_da
from src.utils import timeit
from src.preprocessing.landsat_to_ncf import create_netcdfs
from src.visualisation.ani import animate_prediction


if __name__ == "__main__":
    # usage:  python3 src/models/unet.py > log.txt
    # create_netcdfs() # uncomment to preprocess data.
    cfd = {
        "start_year_i": 0,
        "mid_year_i": 19,
        "end_year_i": 24,
        "take_esa_coords": True,
        "use_ffil": True,
        "use_mfd": False,
    }

    x_da, y_da = return_x_y_da(
        take_esa_coords=cfd["take_esa_coords"],
        use_ffil=cfd["use_ffil"],
        use_mfd=cfd["use_mfd"]
    )  # load preprocessed data from netcdfs
    # there are now 24 years to choose from. 
    # train set goes from 0 to 1. # print(x_da.year.values)
    # test_inversibility()
    x_tr, y_tr = return_xy_np_grid(
        x_da, y_da, year=range(cfd["start_year_i"], cfd["mid_year_i"])
    )  # load numpy train data.
    x_te, y_te = return_xy_np_grid(
        x_da, y_da, year=range(cfd["mid_year_i"], cfd["end_year_i"])
    )  # load numpy test data.

    """
    bst = train_xgb(
        x_tr, compress_esa(y_tr), x_te, compress_esa(y_te)
    )  # train xgboost model.

    
    bst.save_model(
        os.path.join(wandb.run.dir, wandb.run.name + "_xgb.model")
    )  # save model.

    """

    ## wandb.log(cfd)
    x_all, y_all = return_xy_np_grid(
        x_da, y_da, year=range(cfd["start_year_i"], cfd["end_year_i"])
    )  # all data as numpy.

    """


    y_pr_all = decompress_esa(
        bst.predict(xg_all)
    )  # predict whole time period using model

    
    y_pr_da = y_npa_to_xr(
        y_pr_all, y_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"])),
        reshape=False
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
    """
