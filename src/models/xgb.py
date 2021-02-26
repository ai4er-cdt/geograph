"""
xgb.py
======

This module trains an xgboost classification model using preprocessed inputs.

X: Landsat data.
Y: ESA CCI.

Also animates the result.

Usage: 
    python3 src/models/xgb.py 

Currently training parameters are changed inside train_xgb()

TODO: some functions need to be moved to separate files.
TODO: the animation function could be generalised.
TODO: Should add hydra to allow a lot of different model hyperparameters to be passed in.
TODO: Could extend the training/testing data.
"""
import os
import copy
import numpy as np
import numpy.ma as ma
import wandb
import xgboost as xgb
import rioxarray
import dask
import xarray as xr
import rasterio
import time
from functools import wraps
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, WGS84, PREFERRED_CRS
from src.data_loading.landcover_plot_utils import classes_to_rgb
SAT_DIR = "/gws/nopw/j04/ai4er/guided-team-challenge/2021/biodiversity/gee_satellite_data"


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
    @timeit
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
def return_path_dataarray():
    """
    makes path to the google earth engine landsat data.
    if the file doesn't exist, it becomes 'None' in the netcdf.
    """
    incomplete_years = [] # [1984, 1994, 2002, 2003, 2008] 
    # previously these were ignored, now algorithm is robust to absence.
    years = [year for year in range(1984, 2021) if year not in incomplete_years]
    im_type = ["hab", "chern"]
    month_groups = ["JFM", "AMJ", "JAS", "OND"]
    directory = SAT_DIR
    # year, month_group, im_type
    path_array = np.empty([len(years), len(month_groups), len(im_type)], dtype=object)
    for year in years:
        if year not in incomplete_years:
            path = os.path.join(directory, str(year))
            for i in os.listdir(path):
                full_name = os.path.join(path, i)
                coord_list = [years, month_groups, im_type]
                indices = []
                for coord_no, coord in enumerate(coord_list):
                    for counter, value in enumerate(coord):
                        if str(value) in full_name:
                            indices.append(counter)
                path_array[indices[0], indices[1], indices[2]] = full_name
    return xr.DataArray(
                data=path_array,
                dims=["yr", "mn", "ty"],
                coords=dict(
                    yr=years,
                    mn=month_groups,
                    ty=im_type,
                ),
                attrs=dict(
                    description="Paths to tif.",
                ),
            )


@timeit
def return_normalized_array(
    file_name=os.path.join(SAT_DIR, "2012/L7_chern_2012_AMJ.tif"),
    filter_together=True,
    high_limit=3e3,
    low_limit=0,
    high_filter=True,
    low_filter=True,
    common_norm=True,
):
    """
    Function takes name of geotiff and converts it to a preprocessed numpy array.
    TODO: Change names to make it more obvious what's going on.
    :param file_name: full path to .tif image.
    :param filter_together: if True will only display points where all 3 members of a band
            are below the threshold.
    :param high_limit: The aforementioned threshold.
    :param low_limit: Adding a lower threshold.
    :param high_filter: Bool, whether to turn the high limit on.
    :param low_filter: Bool, whether to turn the lower limit on.
    :param common_norm: Bool, whether to norm between the upper and lower limit.
    :return: numpy float array
    """
    # Open the file:
    raster = rasterio.open(file_name)
    
    # Convert to numpy arrays
    if "L8" in file_name :
        # LandSat8 currently stored in different bands for R, G, B.
        red, green, blue = raster.read(2), raster.read(3), raster.read(4)
    else:
        red, green, blue = raster.read(1), raster.read(2), raster.read(3)
    
    # Normalize bands into 0.0 - 1.0 scale
    def norm(array):
        array_min, array_max = np.nanmin(array), np.nanmax(array)
        if common_norm:
            # This doesn't guarantee it's between 0 and 1 if the filter is off.
            return array / (high_limit - low_limit)
        else:
            return (array - array_min) / (array_max - array_min)
    
    def filt(data_array, filter_array):
        return ma.masked_where(filter_array, data_array).filled(np.nan)

    def filter_sep_and_norm(array):
        if high_filter:
            array = filt(array, array >= high_limit).filled(np.nan)
        if low_filter:
            array = filt(array, array <= low_limit).filled(np.nan)
        return norm(array)

    def filter_tog_and_norm(red, green, blue):
        def comb_and_filt(red, green, blue, filter_red, filter_green, filter_blue):
            filter_array = np.logical_or(
                np.logical_or(filter_red, filter_green), filter_blue
            )
            return (
                filt(red, filter_array),
                filt(green, filter_array),
                filt(blue, filter_array),
            )

        if high_filter:
            filter_red, filter_green, filter_blue = (
                red >= high_limit,
                green >= high_limit,
                blue >= high_limit,
            )
            red, green, blue = comb_and_filt(
                red, green, blue, filter_red, filter_green, filter_blue
            )
        if low_filter:
            filter_red, filter_green, filter_blue = (
                red <= low_limit,
                green <= low_limit,
                blue <= low_limit,
            )
            red, green, blue = comb_and_filt(
                red, green, blue, filter_red, filter_green, filter_blue
            )
        return norm(red), norm(green), norm(blue)

    # Normalize band DN
    if not filter_together:
        blue_norm, green_norm, red_norm = (
            filter_sep_and_norm(blue),
            filter_sep_and_norm(green),
            filter_sep_and_norm(red),
        )
    else:
        red_norm, green_norm, blue_norm = filter_tog_and_norm(red, green, blue)
    # Stack bands
    # TODO: Have these been put in the wrong order?
    bgr = np.dstack((blue_norm, green_norm, red_norm))
    # View the color composite
    return bgr


@timeit
def create_netcdfs():
    """
    Create the landsat preprocessed data and save it as netcdfs for the
    different seasons.
    """
    tmp_path = os.path.join(SAT_DIR, "tmp_nc")
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    print(tmp_path)
    path_da = return_path_dataarray()
    for ty, ty_v in enumerate(path_da.coords["ty"].values.tolist()):
        for mn, mn_v in enumerate(path_da.coords["mn"].values.tolist()):
            path_list = []
            for yr in tqdm(range(len(path_da.coords["yr"].values)), ascii=True, desc=ty_v + "  " + mn_v):
                file_name = path_da.isel(yr=yr, mn=mn, ty=ty).values.tolist()
                if file_name != None and os.path.exists(file_name):
                    xr_da = xr.open_rasterio(file_name)
                    data = return_normalized_array(file_name)
                else:
                    file_name = os.path.join(SAT_DIR, "2012/L7_chern_2012_AMJ.tif")
                    xr_da = xr.open_rasterio(file_name)
                    data = return_normalized_array(file_name)  
                    data[:] = np.nan    # make everything nan if the file didn't exist.
                tmp_name = os.path.join(tmp_path, ty_v + "_" + mn_v + "_yr_" + str(path_da.isel(yr=yr).coords["yr"].values) + ".nc")
                xr.DataArray(
                    data=np.expand_dims(data, axis=3),
                    dims=["y", "x", "band", "year"],
                    coords=dict(
                        y=xr_da.coords["y"].values,
                        x=xr_da.coords["x"].values,
                        band=["red", "green", "blue"],
                        year=[path_da.isel(yr=yr).coords["yr"].values],
                        mn=mn_v,
                    ),
                    attrs=dict(
                        description="Normalized reflectance at " + ty_v + ".",
                    ),
                ).astype("float32").to_dataset(name="norm_refl").to_netcdf(tmp_name)
                path_list.append(tmp_name)

            # this option should override chunk size to prevent chunks from being too large.
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                @timeit
                def _cat_ds():
                    print(path_list)
                    return xr.open_mfdataset(path_list, concat_dim="year", chunks={"band":1, "year":1})
                cat_ds = _cat_ds()
                directory = os.path.join(SAT_DIR, "nc_" + ty_v)
                if not os.path.exists(directory):
                    os.mkdir(directory)
                @timeit
                def _save_nc():
                    xr.save_mfdataset([cat_ds], [os.path.join(directory, ty_v + "_" + mn_v + ".nc")])
                _save_nc()


@timeit
def animate_prediction(x_da, y_da, pred_da, video_path="joint_val.mp4"):
    """
    This function animates the inputs, labels, and the corresponding predictions of the model.
    TODO: improve resolution, and make it an input parameter.
    :param x_da: xarray.Dataarray, 3 bands, 4 seasons, 20 years
    :param y_da: xarray.Dataarray, 1 band, 20 years
    :param pred_da: xarray.Dataarray, 1 band, 20 years
    :param video_path: relative text path to output mp4 file.
    Based on code originally from Tom Anderson: tomand@bas.ac.uk.
    """
    def gen_frame_func(x_da, y_da, pred_da, mask=None, mask_type="contour", cmap="pink_r", figsize=15):
        """
        Create imageio frame function for xarray.DataArray visualisation.

        Parameters:
        da (xr.DataArray): Dataset to create video of.

        mask (np.ndarray): Boolean mask with True over masked elements to overlay
        as a contour or filled contour. Defaults to None (no mask plotting).

        mask_type (str): 'contour' or 'contourf' dictating whether the mask is overlaid
        as a contour line or a filled contour.

        """
        cm = copy.copy(plt.get_cmap(cmap))
        cm.set_bad("gray")

        def make_frame(yr):
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))= plt.subplots(3, 2, figsize=(10, 10))
            # da.sel(yr=yr).plot(cmap=cmap, clim=(min, max))

            x_da.isel(yr=yr, mn=0).plot.imshow(ax=ax1)
            x_da.isel(yr=yr, mn=1).plot.imshow(ax=ax2)
            x_da.isel(yr=yr, mn=2).plot.imshow(ax=ax3)
            x_da.isel(yr=yr, mn=3).plot.imshow(ax=ax4)

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlabel('')

            da = xr.DataArray(
                data=classes_to_rgb(y_da.isel(yr=yr).values),
                dims=["y", "x", "band"],
                coords=dict(
                    y=y_da.coords["y"].values,
                    x=y_da.coords["x"].values,
                    band=["red", "green", "blue"],
                    Y='esa_cci',
                ),
            )
            da.plot.imshow(ax=ax5)

            da = xr.DataArray(
                data=classes_to_rgb(np.round_(pred_da.isel(yr=yr)).values.astype('int16')),
                dims=["y", "x", "band"],
                coords=dict(
                    y=y_da.coords["y"].values,
                    x=y_da.coords["x"].values,
                    band=["red", "green", "blue"],
                    Y='predicted_classes',
                ),
            )
            da.plot.imshow(ax=ax6)
            if mask is not None:
                if mask_type == "contour":
                    ax.contour(mask, levels=[0.5, 1], colors="k")
                elif mask_type == "contourf":
                    ax.contourf(mask, levels=[0.5, 1], colors="k")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        return make_frame

    def xarray_to_video(
        x_da, y_da, pred_da,
        video_path,
        fps,
        mask_type="contour",
        video_dates=None,
        cmap="viridis",
        figsize=15,
    ):
        """
        Generate video of an xarray.DataArray. Optionally input a list of
        `video_dates` to show, otherwise the full set of time coordiantes
        of the dataset is used.
        """
        if video_dates is None:
            video_dates = [yr for yr in range(len(y_da.yr.values))]
        make_frame = gen_frame_func(
            x_da=x_da, y_da=y_da, pred_da=pred_da, mask_type=mask_type, cmap=cmap, figsize=figsize
        )
        imageio.mimsave(
            video_path,
            [make_frame(date) for date in tqdm(video_dates, desc=video_path)],
            fps=fps,
        )
        print("Video " + video_path + " made.")

    xarray_to_video(
        x_da, y_da, pred_da,
        video_path,
        mask_type=None,
        fps=5,
    )


@timeit
def _return_x_y_da(
    take_esa_coords=False,
    use_mfd=True,
):
    """
    Load the preprocced Landsat and ESA cci data on the same format of xarray.dataaray grids.
    # TODO: Find a way to memoise this so that it doesn't need to be repeated every time.
    :param take_esa_coords: bool, if true use the lower resolution grid from esa ccis
    current time taken to run: 'return_x_y_da'  431.59196 s
    TODO: Currently when take_esa_coords=False, the process runs out of memory.
    """
    mn_l = ["JFM", "AMJ", "JAS", "OND"]

    @timeit
    def return_y_da():
        input_filepaths = [
                GWS_DATA_DIR / "esa_cci_rois" / f"esa_cci_{year}_chernobyl.geojson"
                for year in range(1992, 2016)
            ]
        da_list = []
        for i in range(len(input_filepaths)):
            file_name = input_filepaths[i]
            da_list.append(xr.open_rasterio(file_name).isel(band=0))
        return xr.concat(da_list, "yr").assign_coords(yr=("yr", list(range(1992, 2016))))

    @timeit
    def return_part_x_da(ty_v="chern", mn_v="JAS"):
        directory = os.path.join(SAT_DIR, "nc_"+ ty_v)
        return xr.open_dataset(os.path.join(directory, ty_v + "_" + mn_v + ".nc")).norm_refl
    
    def return_x_name_list(ty_v="chern"):
        directory = os.path.join(SAT_DIR, "nc_"+ ty_v)
        return [os.path.join(directory, ty_v + "_" + mn_v + ".nc") for mn_v in mn_l],
    
    @timeit
    def reindex_da(mould_da, putty_da):
        putty_da.reindex(
                x=mould_da.coords["x"].values,
                y=mould_da.coords["y"].values,
                method="nearest",
            )

    if take_esa_coords:
        y_full_da = return_y_da()
        x_full_da = xr.concat(
            [reindex_da(y_full_da, return_part_x_da(mn_v=mn_v)) for mn_v in mn_l],
            "mn").assign_coords(mn=("mn", mn_l))
    else:
        if not use_mfd:
            x_full_da = xr.concat(
                [return_part_x_da(mn_v=mn_v) for mn_v in mn_l],
                "mn").assign_coords(mn=("mn", mn_l))
        else:
            print(return_x_name_list(ty_v="chern"))
            x_full_da= xr.open_mfdataset(return_x_name_list(ty_v="chern"), 
                        concat_dim="mn", chunks={"yr": 1}).norm_refl.assign_coords(mn=("mn", mn_l))
        y_full_da = reindex_da(x_full_da, return_y_da())
        
    def intersection(lst1, lst2):
        return [value for value in lst1 if value in lst2]

    y_years = y_full_da.coords["yr"].values.tolist()
    x_years = x_full_da.coords["yr"].yr.to_dict()["coords"]["year"]["data"]
    int_years = intersection(y_years, x_years)
    x_yr_i = [x_years.index(x) for x in int_years];  y_yr_i = [y_years.index(x) for x in int_years]
    return x_full_da.isel(yr=x_yr_i), y_full_da.isel(yr=y_yr_i)


@timeit
def return_x_y_da(
    take_esa_coords=False,
    use_mfd=True,
    ):
    names = ["take_esa_coords_" + str(take_esa_coords) + "_use_mfd_" 
             + str(use_mfd) + "_" + v + ".nc" for v in ["x", "y"]]
    direc = os.path.join(SAT_DIR, "inputs")
    if not os.path.exists(direc):
        os.mkdir(direc)
    full_names = [os.path.join(direc, name) for name in names]
    if not os.path.exists(full_names[0]) or not os.path.exists(full_names[0]):
        x_da, y_da = _return_x_y_da(take_esa_coords=take_esa_coords, use_mfd=use_mfd)
        if use_mfd:
            xr.save_mfdataset([y_da.to_dataset()], [full_names[0]])
            xr.save_mfdataset([y_da.to_dataset()], [full_names[1]])
        else:
            x_da.to_netcdf(full_names[0])
            y_da.to_netcdf(full_names[1])
    else:
        if use_mfd:
            x_da = xr.open_mfdataset([full_names[0]])
            y_da = xr.open_mfdataset([full_names[1]])
        else:
            x_da = xr.open_dataset(full_names[0]).to_dataarray()
            y_da = xr.open_dataset(full_names[1]).to_dataarray()
    return x_da, y_da


@timeit
def return_xy_npa(x_da, y_da, yr=5):
    """
    return the x and y numpy arrays for a given number of years.
    :param x_da: xarray.dataarray, inputs
    :param y_da: xarray.dataarray, labels
    :param yr: ints, single or list
    :return: x_val, y_val
    """
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
def npa_to_xarray(npa, da):
    """
       Reformat numpy array to be like a given xarray.dataarray. 
       Inverse of return_xy for the y values at least.
       :param npa: numpy array, float.
       :param da: xarray.dataarray, the mould da for the output.
       :return: xarray.dataarray, containing npa.
    """
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
    Train an xgboost model using numpy inputs.
    :param train_X: npa, float32
    :param train_Y: npa, int16
    :param test_X: npa, float32
    :param test_Y: npa, int16
    TODO: Make mapping between esa_cci and a reduced list of labels.
    """
    wandb.init(project="xgbc-esa-cci", entity="sdat2")   # my id for wandb
    # label need to be 0 to num_class -1
    xg_train = xgb.DMatrix(train_X, label=train_Y)     # make train DMatrix
    xg_test = xgb.DMatrix(test_X, label=test_Y)        # make test DMatrix
    # setup parameters for xgboost
    param = {}
    param["objective"] = "multi:softmax"      # use softmax multi-class classification
    param["eta"] = 0.1                        # scale weight of positive examples
    param["max_depth"] = 6                    # max_depth
    param["silent"] = 1
    param["nthread"] = 16                     # number of threads
    param['num_class'] = np.max(train_Y) + 1  # max size of labels.
    wandb.config.update(param)
    watchlist = [(xg_train, "train"), (xg_test, "test")]
    num_round = 15  # how many training epochs
    bst = xgb.train(
        param,
        xg_train,
        num_round,
        watchlist,
        callbacks=[wandb.xgboost.wandb_callback()],
    )
    # get prediction
    pred = bst.predict(xg_test)                                 # predict for the test set.
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    print("Test error using softmax = {}".format(error_rate))
    wandb.log({"Error Rate": error_rate})
    return bst


if __name__ == "__main__":
    # usage:  python3 src/models/xgb.py > log.txt
    create_netcdfs() # uncomment to preprocess data.
    run_name = "6-MS"                                      # memorable run-name for saving.
    direc = run_name
    if not os.path.exists(direc):                          # check if the direc exists.
        os.mkdir(direc)                                    # make the directory if it doesn't exist.
    x_da, y_da = return_x_y_da()                           # load preprocessed data from netcdfs
    print(x_da.yr.values)
    x_tr, y_tr = return_xy_npa(x_da, y_da, yr=range(0, 15))    # load numpy training data 
    x_te, y_te = return_xy_npa(x_da, y_da, yr=range(15, 20))   # load numpy test data
    bst = train_xgb(x_tr, y_tr, x_te, y_te)                # train xgboost model
    x_all, y_all = return_xy_npa(x_da, y_da, yr=range(0, 20))  # load all data as numpy.
    xg_all = xgb.DMatrix(x_all, label=y_all)               # pass all data to xgb data matrix
    y_pr_all = bst.predict(xg_all)                         # predict whole time period using model
    y_pr_da = npa_to_xarray(y_pr_all, y_da)                    # transform full prediction to dataarray.
    y_pr_da.to_netcdf(os.path.join(direc, run_name  + "_y.nc"))     # save to netcdf
    bst.save_model(os.path.join(direc, run_name + "_xgb.model"))     # save model using xgboost
    animate_prediction(x_da, y_da, y_pr_da, 
                       video_path=os.path.join(direc, 
                                   run_name + "_joint_val.mp4"))  # animate prediction vs inputs.