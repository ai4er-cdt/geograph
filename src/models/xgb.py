import os
import numpy as np
import wandb
import xgboost as xgb
import rioxarray
import xarray as xr
import rasterio
import time
from functools import wraps
from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, WGS84, PREFERRED_CRS


def timeit(method):
    """
    timeit is a wrapper for performance analysis which should
    return the time taken for a function to run,
    :param method: the function that it takes as an input
    :return: timed
    example usage:
    tmp_log_data={}
    part = spin_forward(400, co, particles=copy.deepcopy(particles),
                        log_time=tmp_log_d)
    # chuck it into part to stop interference.
    assert part != particles
    spin_round_time[key].append(tmp_log_data['SPIN_FORWARD'])
    USAGE:
    @ttimeit
    """

    @wraps(method)
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = te - ts
        else:
            print("%r  %2.5f s\n" % (method.__name__, (te - ts)))
        return result

    return timed


@timeit
def return_x_y_da():
    input_filepaths = [
        GWS_DATA_DIR / "esa_cci_rois" / f"esa_cci_{year}_chernobyl.geojson"
        for year in range(1992, 2016)
    ]

    da_list = []

    for i in range(len(input_filepaths)):
        file_name = input_filepaths[i]
        da_list.append(xr.open_rasterio(file_name).isel(band=0))

    y_old_da = xr.concat(da_list, "yr")
    y_old_da = y_old_da.assign_coords(yr=("yr", list(range(1992, 2016))))

    def return_da(ty_v="chern", mn_v="JAS"):
        directory = (
            "/gws/nopw/j04/ai4er/guided-team-challenge/2021/biodiversity/gee_satellite_data/nc_"
            + ty_v
        )
        return xr.open_dataarray(os.path.join(directory, ty_v + "_" + mn_v + ".nc"))

    mn_l = ["JFM", "AMJ", "JAS", "OND"]
    x_old_da = xr.concat(
        [
            return_da(mn_v=x).reindex(
                x=y_old_da.coords["x"].values,
                y=y_old_da.coords["y"].values,
                method="nearest",
            )
            for x in mn_l
        ],
        "mn",
    ).assign_coords(mn=("mn", mn_l))

    def intersection(lst1, lst2):
        return [value for value in lst1 if value in lst2]

    y_years = y_old_da.coords["yr"].values.tolist()
    x_years = x_old_da.coords["yr"].yr.to_dict()["coords"]["year"]["data"]
    int_years = intersection(y_years, x_years)
    x_indices = [x_years.index(x) for x in int_years]
    y_indices = [y_years.index(x) for x in int_years]

    x_da = x_old_da.isel(yr=x_indices)
    y_da = y_old_da.isel(yr=y_indices)

    return x_da, y_da


@timeit
def return_xy(x_da, y_da, yr=5):
    x_val = np.asarray(
        [
            x_da.isel(yr=yr, mn=mn, band=band).values.ravel()
            for mn in range(4)
            for band in range(3)
        ]
    )
    x_val = np.swapaxes(x_val, 0, 1)
    y_val = y_da.isel(yr=yr).values.ravel()
    return x_val, y_val


@timeit
def to_netcdf(npa, da):
    x = da.x.values
    y = da.y.values
    yr = da.yr.values
    return xr.DataArray(
        data=npa.reshape(da.values.shape),
        dims=da.dims,
        coords=dict(
            x=(["x"], x),
            y=(["y"], y),
            yr=yr,
        ),
    )


@timeit
def train_xgb(train_X, train_Y, test_X, test_Y):
    """
    :param train_X: npa, float32
    :param train_Y: npa, int16
    :param test_X: npa, float32
    :param test_Y: npa, int16
    TODO: Make mapping between esa_cci and a reduced list of labels.
    """
    wandb.init(project="xgbc-esa-cci", entity="sdat2")
    # label need to be 0 to num_class -1
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param["objective"] = "multi:softmax"
    # scale weight of positive examples
    param["eta"] = 0.1
    param["max_depth"] = 8
    param["silent"] = 1
    param["nthread"] = 16
    param['num_class'] = np.max(train_Y) + 1 # I'm not sure I want to set this.
    wandb.config.update(param)
    watchlist = [(xg_train, "train"), (xg_test, "test")]
    num_round = 15
    bst = xgb.train(
        param,
        xg_train,
        num_round,
        watchlist,
        callbacks=[wandb.xgboost.wandb_callback()],
    )
    # get prediction
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print("Test error using softmax = {}".format(error_rate))
    wandb.summary["Error Rate"] = error_rate
    return bst


if __name__ == "__main__":
    # python3 src/models/xgb.py > log.txt
    run_name = "8-DEEP"
    x_da, y_da = return_x_y_da()
    x_tr, y_tr = return_xy(x_da, y_da, yr=range(0, 15))
    x_te, y_te = return_xy(x_da, y_da, yr=range(15, 20))
    bst = train_xgb(x_tr, y_tr, x_te, y_te)
    x_all, y_all = return_xy(x_da, y_da, yr=range(0, 20))
    xg_all = xgb.DMatrix(x_all, label=y_all)
    y_pr_all = bst.predict(xg_all)
    y_pr_da = to_netcdf(y_pr_all, y_da)
    y_pr_da.to_netcdf(run_name + "_y.nc")
    bst.save_model(run_name + "_xgb.model")
