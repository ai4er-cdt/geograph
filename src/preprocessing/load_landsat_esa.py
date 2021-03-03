"""
load_landsat_esa.py
================================================================

from  src.preprocessing.load_landsat_esa import return_xy_npa, y_npa_to_xarray, x_npa_to_xarray

"""
import os
import numpy as np
import xarray as xr
from src.constants import ESA_LANDCOVER_DIR, GWS_DATA_DIR, SAT_DIR
from src.preprocessing.esa_comp import compress_esa, decompress_esa, FORW_D, REV_D
from src.utils import timeit
from src.preprocessing.landsat import create_netcdfs
from src.visualisation.ani import animate_prediction


@timeit
def test_inversibility(x_da, y_da, cfd):
        # run-20210225_161033-yxyit68w/
    x_all, y_all = return_xy_npa(
        x_da, y_da, year=range(cfd["start_year_i"], cfd["end_year_i"])
        )  # all data as numpy.
    x_rp, y_rp = return_xy_npa(
    x_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"])),
    y_npa_to_xarray(
                y_all, y_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"]))
            ),
            year=range(cfd["start_year_i"], cfd["end_year_i"]),
        )
    assert np.all(y_all == y_rp)


@timeit
def _return_x_y_da(
    take_esa_coords=False,
    use_mfd=True,
    use_ffil=True,
):
    """
    This function is memoised by return_x_y_da() so that if it's already been run with
    the same inputs, it will load the preprocessed data.
    Load the preprocced Landsat and ESA-CCI data on the same format of xarray.dataaray.
    :param take_esa_coords: bool, if true use the lower resolution grid from esa ccis
    :param use_mfd: use mfd to load datasets so that lazy loading / computation is achieved.
    :param use_ffil: forward fill nan values along dim year.
    current time taken to run: '_return_x_y_da'  431.59196 s
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
        return xr.concat(da_list, "year").assign_coords(
            year=("year", list(range(1992, 2016)))
        )

    @timeit
    def return_part_x_da(ty_v="chern", mn_v="JAS"):
        directory = os.path.join(SAT_DIR, "nc_" + ty_v)
        return xr.open_dataset(
            os.path.join(directory, ty_v + "_" + mn_v + ".nc")
        ).norm_refl

    def return_x_name_list(ty_v="chern"):
        directory = os.path.join(SAT_DIR, "nc_" + ty_v)
        return ([os.path.join(directory, ty_v + "_" + mn_v + ".nc") for mn_v in mn_l],)

    @timeit
    def reindex_da(mould_da, putty_da):
        """reindex the putty_da to become like the mould_da"""
        return putty_da.reindex(
            x=mould_da.coords["x"].values,
            y=mould_da.coords["y"].values,
            method="nearest",
        )
    
    @timeit
    def ffil(da, dim="year"):
        return da.ffill(dim) 

    if take_esa_coords:
        y_full_da = return_y_da()
        x_full_da = xr.concat(
            [reindex_da(y_full_da, return_part_x_da(mn_v=mn_v)) for mn_v in mn_l], "mn"
        ).assign_coords(mn=("mn", mn_l))
    else:
        if not use_mfd:
            x_full_da = xr.concat(
                [return_part_x_da(mn_v=mn_v) for mn_v in mn_l], "mn"
            ).assign_coords(mn=("mn", mn_l))
        else:
            print(return_x_name_list(ty_v="chern"))
            x_full_da = xr.open_mfdataset(
                return_x_name_list(ty_v="chern"), concat_dim="mn", chunks={"year": 1},
                lock=False
            ).norm_refl.assign_coords(mn=("mn", mn_l))
        y_full_da = reindex_da(x_full_da, return_y_da())

    def intersection(lst1, lst2):
        return [value for value in lst1 if value in lst2]

    y_years = y_full_da.coords["year"].values.tolist()
    x_years = x_full_da.coords["year"].values.tolist()
    int_years = intersection(y_years, x_years)
    x_year_i = [x_years.index(x) for x in int_years]
    y_year_i = [y_years.index(x) for x in int_years]

    if use_ffil:
        return ffil(x_full_da).isel(year=x_year_i), y_full_da.isel(year=y_year_i)
    else:
        return x_full_da.isel(year=x_year_i), y_full_da.isel(year=y_year_i)


@timeit
def return_x_y_da(
    take_esa_coords=False,
    use_mfd=True,
    use_ffil=False,
):
    """
    Uses _return_x_y_da() only if the netcdf has not already been made.
    :param take_esa_coords: lower resolution
    :param use_mfd: use mfd
    """
    names = [
        "take_esa_coords_"
        + str(take_esa_coords)
        + "_use_mfd_"
        + str(use_mfd)
        + "_use_ffil_"
        + str(use_ffil)
        + "_"
        + v
        + ".nc"
        for v in ["x", "y"]
    ]
    direc = os.path.join(SAT_DIR, "inputs")
    if not os.path.exists(direc):
        os.mkdir(direc)
    full_names = [os.path.join(direc, name) for name in names]
    if (not os.path.exists(full_names[0])) or (not os.path.exists(full_names[1])):
        print("x/y values not discovered. Remaking them.")
        x_da, y_da = _return_x_y_da(take_esa_coords=take_esa_coords, use_mfd=use_mfd, use_ffil=use_ffil)
        x_ds, y_ds = x_da.to_dataset(name="norm_refl"), y_da.to_dataset(name="esa_cci")
        if use_mfd:
            xr.save_mfdataset([x_ds], [full_names[0]])
            xr.save_mfdataset([y_ds], [full_names[1]])
        else:
            x_ds.to_netcdf(full_names[0])
            y_ds.to_netcdf(full_names[1])
    else:
        print("x/y values premade. Reusing them.")
        if use_mfd:
            x_ds = xr.open_mfdataset([full_names[0]], chunks={"year": 1}, lock=False)
            y_ds = xr.open_mfdataset([full_names[1]], chunks={"year": 1}, lock=False)
        else:
            x_ds = xr.open_dataset(full_names[0])
            y_ds = xr.open_dataset(full_names[1])
    
    return x_ds.norm_refl, y_ds.esa_cci


@timeit
def return_xy_npa(x_da, y_da, year=5):
    """
    return the x and y numpy arrays for a given number of years.
    Currently this function just returns (N, D) for x and (N,) for Y
    for UNET we want a function that returns (yr, y, xr, D) for x and (yr, y, x, D) for y
    :param x_da: xarray.dataarray, inputs
    :param y_da: xarray.dataarray, labels
    :param year: ints, single or list
    :return: x_val, y_val
    """

    def combine_first_two_indices(x_val, y_val):
        return (
            np.swapaxes(
                np.array([x_val[:, :, i].ravel() for i in range(x_val.shape[2])]), 0, 1
            ),
            y_val.ravel(),
        )

    def _return_xy_npa(x_da, y_da, yr=5):
        assert isinstance(yr, int)
        x_val = np.asarray(
            [
                x_da.isel(year=yr, mn=mn, band=band).values.ravel()
                for mn in range(4)
                for band in range(3)
            ]
        )
        # [mn, band]
        return np.swapaxes(x_val, 0, 1), y_da.isel(year=yr).values.ravel()

    if isinstance(year, range) or isinstance(year, list):
        x_val_l, y_val_l = [], []
        for yr in year:
            x_val_p, y_val_p = _return_xy_npa(x_da, y_da, yr=yr)
            x_val_l.append(x_val_p)
            y_val_l.append(y_val_p)
        x_val, y_val = combine_first_two_indices(np.array(x_val_l), np.array(y_val_l))
    else:
        x_val, y_val = _return_xy_npa(x_da, y_da, yr=year)
    return x_val, y_val


@timeit
def y_npa_to_xarray(npa, da):
    """
    Reformat numpy array to be like a given xarray.dataarray.
    Inverse of return_xy for the y values at least.
    :param npa: numpy array, float.
    :param da: xarray.dataarray, the mould da for the output.
    :return: xarray.dataarray, containing npa.
    """
    x = da.x.values
    y = da.y.values
    coords_d = dict(x=(["x"], x), y=(["y"], y))
    coords_d["year"] = da.year.values
    return xr.DataArray(
        data=npa.reshape(da.values.shape),
        dims=da.dims,
        coords=coords_d,
    )


@timeit
def x_npa_to_xarray(npa, da):
    """
    Reformat numpy array to be like a given xarray.dataarray.
    :param npa: numpy array, float.
    :param da: xarray.dataarray, the mould da for the output.
    :return: xarray.dataarray, containing npa.
    """
    map_to_feat = np.array([[mn, band] for mn in range(4) for band in range(3)])
    n_l = []
    for i in range(map_to_feat.shape[0]):
        if len(n_l) <= map_to_feat[i][0]:
            n_l.append([])
        n_l[-1].append(
            npa[:, i].reshape((len(da.year.values), len(da.y.values), len(da.x.values)))
        )

    dims = ("mn", "band", "year", "y", "x")
    coords_d = {}
    for dim in dims:
        coords_d[dim] = da.coords[dim].values

    return xr.DataArray(
        data=np.array(n_l),
        dims=dims,
        coords=coords_d,
    )
