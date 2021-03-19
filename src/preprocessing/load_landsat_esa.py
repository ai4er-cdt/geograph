# pylint: disable-all
"""
load_landsat_esa.py
================================================================

usage:

from src.preprocessing.load_landsat_esa import return_xy_npa, y_npa_to_xarray,
x_npa_to_xarray

"""
import os
import time
from typing import Sequence, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from src.constants import GWS_DATA_DIR, SAT_DIR  # ESA_LANDCOVER_DIR,
from src.utils import timeit

# import dask
# from dask.distributed import Client
# from src.preprocessing.esa_compress import compress_esa, decompress_esa,
# FORW_D, REV_D


@timeit
def return_x_y_da(
    take_esa_coords: bool = False,
    use_mfd: bool = True,
    use_ffil: bool = False,
    use_ir: bool = False,
    prefer_remake: bool = False,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Returns compatible X and Y xr.datarrays for Chernobyl
        can take roughly  1106 s if making full dataset.
        Memoises function _return_x_y_da so that the dataset doesn't need to
        be remade if it already exists with the same parameters.

    Args:
        take_esa_coords (bool, optional): If true maps onto 300m ESA CCI grid.
        Defaults to False.
        use_mfd (bool, optional): Use open_mfdataset etc. for ultra-lazy loading.
        Defaults to True.
        use_ffil (bool, optional): Fill forward Landsat to remove nans.
        Defaults to False.
        use_ir (bool, optional): Take Landsat IR bands. Defaults to False.
        prefer_remake (bool, optional): Will ignore existing files, and remake
        them instead. Defaults to False.

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: X and Y xarray.DataArray's
    """
    names = [
        "take_esa_coords_"
        + str(take_esa_coords)
        + "_use_mfd_"
        + str(use_mfd)
        + "_use_ffil_"
        + str(use_ffil)
        + "_use_ir_"
        + str(use_ir)
        + "_"
        + v
        + ".nc"
        for v in ["x", "y"]
    ]
    direc = os.path.join(SAT_DIR, "inputs")
    if not os.path.exists(direc):
        os.mkdir(direc)
    full_names = [os.path.join(direc, name) for name in names]
    print(full_names)
    if (
        (not os.path.exists(full_names[0]))
        or (not os.path.exists(full_names[1]))
        or prefer_remake
    ):
        print("x/y values not reused. Remaking them.")
        x_da, y_da = _return_x_y_da(
            take_esa_coords=take_esa_coords,
            use_mfd=use_mfd,
            use_ffil=use_ffil,
            use_ir=use_ir,
        )
        print(x_da)
        print(y_da)
        x_ds, y_ds = x_da.to_dataset(name="norm_refl"), y_da.to_dataset(name="esa_cci")

        if use_mfd:
            print("saving x values")
            xr.save_mfdataset([x_ds], [full_names[0]])
            print("saving y values")
            xr.save_mfdataset([y_ds], [full_names[1]])
            print("saving all values")
        else:
            print("saving x values")
            x_ds.to_netcdf(full_names[0])
            print("saving y Values")
            y_ds.to_netcdf(full_names[1])
            print("saving all values")
    else:
        print("x/y values premade. Reusing them.")
        if use_mfd:
            x_ds = xr.open_mfdataset([full_names[0]], chunks={"year": 1}, lock=False)
            y_ds = xr.open_mfdataset([full_names[1]], chunks={"year": 1}, lock=False)
            x_ds, y_ds = clip(x_ds, y_ds)
        else:
            x_ds = xr.open_dataset(full_names[0])
            y_ds = xr.open_dataset(full_names[1])
            x_ds, y_ds = clip(x_ds, y_ds)
    return x_ds.norm_refl, y_ds.esa_cci


@timeit
def _return_x_y_da(
    take_esa_coords: bool = False,
    use_mfd: bool = True,
    use_ffil: bool = True,
    use_ir: bool = False,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """This function is memoised by return_x_y_da() so that if it's already
       been run with the same inputs, it won't run again.

    Args:
        take_esa_coords (bool, optional): if true use the lower resolution grid
        from esa ccis. Defaults to False.
        use_mfd (bool, optional): lazy loading / computation. Defaults to True.
        use_ffil (bool, optional): forward fill nan values along dim year.
        Defaults to True.
        use_ir (bool, optional): use Landsat IR bands. Defaults to False.

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: X and Y xarray.DataArray's
    """
    mn_l = ["JFM", "AMJ", "JAS", "OND"]

    @timeit
    def return_y_da() -> xr.DataArray:
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
    def return_part_x_da(
        ty_v: str = "chern", mn_v: str = "JAS", ir_ap: str = ""
    ) -> xr.DataArray:
        directory = os.path.join(SAT_DIR, "nc_" + ty_v)
        print(os.path.join(directory, ty_v + "_" + mn_v + ir_ap + ".nc"))
        return xr.open_dataset(
            os.path.join(directory, ty_v + "_" + mn_v + ir_ap + ".nc")
        ).norm_refl

    def return_x_name_list(ty_v: str = "chern") -> Sequence[str]:
        directory = os.path.join(SAT_DIR, "nc_" + ty_v)
        return ([os.path.join(directory, ty_v + "_" + mn_v + ".nc") for mn_v in mn_l],)

    @timeit
    def reindex_da(
        mould_da: xr.DataArray, putty_da: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """reindex the putty_da to become like the mould_da"""
        return putty_da.reindex(
            x=mould_da.coords["x"].values,
            y=mould_da.coords["y"].values,
            method="nearest",
        )

    y_full_da = return_y_da()
    if take_esa_coords:
        if use_ir:
            ts = time.perf_counter()
            x_full_da = xr.concat(
                [
                    xr.concat(
                        [
                            reindex_da(y_full_da, return_part_x_da(mn_v=mn_v))
                            for mn_v in mn_l
                        ],
                        "mn",
                    ).assign_coords(mn=("mn", mn_l)),
                    xr.concat(
                        [
                            reindex_da(
                                y_full_da, return_part_x_da(mn_v=mn_v, ir_ap="_IR")
                            )
                            for mn_v in mn_l
                        ],
                        "mn",
                    ).assign_coords(mn=("mn", mn_l)),
                ],
                "band",
            )
            te = time.perf_counter()
            print("time for concats %2.5f s\n" % (te - ts))
            x_full_da, y_full_da = clip(x_full_da, y_full_da)
            print("made x da")
        else:
            x_full_da = xr.concat(
                [reindex_da(y_full_da, return_part_x_da(mn_v=mn_v)) for mn_v in mn_l],
                "mn",
            ).assign_coords(mn=("mn", mn_l))
            x_full_da, y_full_da = clip(x_full_da, y_full_da)
    else:
        if not use_mfd:
            if use_ir:
                x_full_vis = xr.concat(
                    [return_part_x_da(mn_v=mn_v, ir_ap="") for mn_v in mn_l], "mn"
                ).assign_coords(mn=("mn", mn_l))
                print("Visible bands read")
                x_full_ir = xr.concat(
                    [return_part_x_da(mn_v=mn_v, ir_ap="_IR") for mn_v in mn_l], "mn"
                ).assign_coords(mn=("mn", mn_l))
                print("IR bands read")
                x_full_da = xr.concat([x_full_vis, x_full_ir], "band")
                print("all bands merged")
            else:
                x_full_da = xr.concat(
                    [return_part_x_da(mn_v=mn_v) for mn_v in mn_l], "mn"
                ).assign_coords(mn=("mn", mn_l))
        else:
            print(return_x_name_list(ty_v="chern"))
            x_full_da = xr.open_mfdataset(
                return_x_name_list(ty_v="chern"),
                concat_dim="mn",
                chunks={"year": 1},
                lock=False,
            ).norm_refl.assign_coords(mn=("mn", mn_l))
        x_full_da, y_full_da = clip(x_full_da, y_full_da)
        y_full_da = reindex_da(x_full_da, y_full_da)

    def intersection(lst1: list, lst2: list) -> list:
        return [value for value in lst1 if value in lst2]

    y_years = y_full_da.coords["year"].values.tolist()
    x_years = x_full_da.coords["year"].values.tolist()
    int_years = intersection(y_years, x_years)
    x_year_i = [x_years.index(x) for x in int_years]
    y_year_i = [y_years.index(x) for x in int_years]

    @timeit
    def ffil(da: xr.DataArray, dim: str = "year") -> xr.DataArray:
        return da.ffill(dim)

    if use_ffil:
        return ffil(x_full_da).isel(year=x_year_i), y_full_da.isel(year=y_year_i)
    else:
        return x_full_da.isel(year=x_year_i), y_full_da.isel(year=y_year_i)


@timeit
def clip(da_1: xr.DataArray, da_2: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """Mutually clips the dataarrays so that they end up within the same window.

    Args:
        da_1 (xr.DataArray): y and x coords comparible to da_2.
        da_2 (xr.DataArray): y and x coords comparible to da_2.

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: y and x coords compatible between
        da_1 and da_2.
    """
    y_lim = [
        max([da_1.y.min(), da_2.y.min()]).values.tolist(),
        min([da_1.y.max(), da_2.y.max()]).values.tolist(),
    ]
    x_lim = [
        max([da_1.x.min(), da_2.x.min()]).values.tolist(),
        min([da_1.x.max(), da_2.x.max()]).values.tolist(),
    ]

    def isAscending(xs):
        for n in range(len(xs) - 1):
            if xs[n] > xs[n + 1]:
                return False
        return True

    def isDescending(xs):
        for n in range(len(xs) - 1):
            if xs[n] < xs[n + 1]:
                return False
        return True

    if isDescending(da_1.x.values.tolist()):
        da_1 = da_1.sel(x=slice(x_lim[1], x_lim[0]))
    elif isAscending(da_1.x.values.tolist()):
        da_1 = da_1.sel(x=slice(x_lim[0], x_lim[1]))
    else:
        assert False

    if isDescending(da_2.x.values.tolist()):
        da_2 = da_2.sel(x=slice(x_lim[1], x_lim[0]))
    elif isAscending(da_2.x.values.tolist()):
        da_2 = da_2.sel(x=slice(x_lim[0], x_lim[1]))
    else:
        assert False

    if isDescending(da_1.y.values.tolist()):
        da_1 = da_1.sel(y=slice(y_lim[1], y_lim[0]))
    elif isAscending(da_1.y.values.tolist()):
        da_1 = da_1.sel(y=slice(y_lim[0], y_lim[1]))
    else:
        assert False

    if isDescending(da_2.y.values.tolist()):
        da_2 = da_2.sel(y=slice(y_lim[1], y_lim[0]))
    elif isAscending(da_2.y.values.tolist()):
        da_2 = da_2.sel(y=slice(y_lim[0], y_lim[1]))
    else:
        assert False

    return da_1, da_2


@timeit
def return_xy_npa(
    x_da: xr.DataArray, y_da: xr.DataArray, year=5
) -> Tuple[np.ndarray, np.ndarray]:
    """return the x and y numpy arrays for a given number of years.
    Currently this function just returns (N, D) for x and (N,) for Y

    Args:
        x_da (xr.DataArray): inputs
        y_da (xr.DataArray): labels
        year (int, optional): single or list. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x_val, y_val
    """

    def combine_first_two_indices(
        x_val: np.ndarray, y_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.swapaxes(
                np.array([x_val[:, :, i].ravel() for i in range(x_val.shape[2])]), 0, 1
            ),
            y_val.ravel(),
        )

    def _return_xy_npa(
        x_da: xr.DataArray, y_da: xr.DataArray, yr: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(yr, int)
        x_val = np.asarray(
            [
                x_da.isel(year=yr, mn=mn, band=band).values.ravel()
                for mn in range(len(x_da.mn.values))
                for band in range(len(x_da.band.values))
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
def return_xy_dask(
    x_da: xr.DataArray, y_da: xr.DataArray, year: int = 5
) -> Tuple[da.array, da.array]:
    """return the x and y numpy arrays for a given number of years.
    Currently this function just returns (N, D) for x and (N,) for Y

    Args:
        x_da (xr.DataArray): inputs
        y_da (xr.DataArray): labels
        year (int, optional): single or list. Defaults to 5.

    Returns:
        Tuple[da.array, da.array]: x_val, y_val
    """

    def combine_first_two_indices(
        x_val: da.array, y_val: da.array
    ) -> Tuple[da.array, da.array]:
        return (
            da.stack([x_val[:, :, i].ravel() for i in range(x_val.shape[2])], axis=1),
            y_val.ravel(),
        )

    def _return_xy_dask_array(
        x_da: xr.DataArray, y_da: xr.DataArray, yr: int = 5
    ) -> Tuple[da.array, da.array]:
        assert isinstance(yr, int)
        x_val = da.stack(
            [
                x_da.isel(year=0, mn=mn, band=band).data.ravel()
                for mn in range(len(x_da.mn.values))
                for band in range(len(x_da.band.values))
            ],
            axis=1,
        )  # [mn, band]
        return x_val, y_da.isel(year=yr).data.ravel()

    if isinstance(year, range) or isinstance(year, list):
        x_val_l, y_val_l = [], []
        for yr in year:
            x_val_p, y_val_p = _return_xy_dask_array(x_da, y_da, yr=yr)
            x_val_l.append(x_val_p)
            y_val_l.append(y_val_p)
        # x_val, y_val = da.stack(x_val_l), da.stack(y_val_l)
        x_val, y_val = combine_first_two_indices(da.stack(x_val_l), da.stack(y_val_l))
    else:
        x_val, y_val = _return_xy_dask_array(x_da, y_da, yr=year)
    return x_val, y_val


@timeit
def y_npa_to_xr(
    npa: np.ndarray, xda: xr.DataArray, reshape: bool = True
) -> xr.DataArray:
    """Reformat numpy array to be like a given xarray.dataarray.
       Inverse of return_xy for the y values at least.

    Args:
        npa (np.array): float array for output
        xda (xr.DataArray): the mould DataArray for the output.
        reshape (bool, optional): Whether or not to reshape the npa.
           Defaults to True.

    Returns:
        xr.DataArray: containing npa in output
    """
    x = xda.x.values
    y = xda.y.values
    coords_d = dict(x=(["x"], x), y=(["y"], y))
    coords_d["year"] = xda.year.values

    if reshape:
        data = npa.reshape(xda.values.shape)
    else:
        data = npa

    return xr.DataArray(
        data=data,
        dims=xda.dims,
        coords=coords_d,
    )


@timeit
def x_npa_to_xr(npa: np.ndarray, da: xr.DataArray) -> xr.DataArray:
    """Reformat numpy array to be like a given xarray.dataarray.

    Args:
        npa (np.array): numpy array, float.
        da (xr.DataArray): xarray.DataArray, the mould da for the output.

    Returns:
        xr.DataArray:  xarray.DataArray, containing npa.
    """
    map_to_feat = np.array(
        [
            [mn, band]
            for mn in range(len(da.mn.values))
            for band in range(len(da.band.values))
        ]
    )
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


@timeit
def return_xy_np_grid(
    x_da: xr.DataArray, y_da: xr.DataArray, year: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """return the x and y numpy arrays for a given number of years.
       for UNET we want a function that returns (yr, y, xr, D) for x
       and (yr, y, x, D) for y

    Args:
        x_da (xr.DataArray): inputs
        y_da (xr.DataArray): labels
        year (int, optional): single or list. Defaults to 5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x_val, y_val
    """

    def _return_xy_npa(x_da, y_da, yr=5):
        assert isinstance(yr, int)
        x_val = np.swapaxes(
            np.swapaxes(
                np.asarray(
                    [
                        x_da.isel(year=yr, mn=mn, band=band).values  # .ravel()
                        for mn in range(len(x_da.mn.values))
                        for band in range(len(x_da.band.values))
                    ]
                ),
                0,
                1,
            ),
            1,
            2,
        )
        # x, y, z
        return x_val, y_da.isel(year=yr).values  # .ravel()

    if isinstance(year, range) or isinstance(year, list):
        x_val_l, y_val_l = [], []
        for yr in year:
            x_val_p, y_val_p = _return_xy_npa(x_da, y_da, yr=yr)
            x_val_l.append(x_val_p)
            y_val_l.append(y_val_p)
        x_val, y_val = np.array(x_val_l), np.array(y_val_l)
    else:
        x_val, y_val = _return_xy_npa(x_da, y_da, yr=year)
    return x_val, y_val


@timeit
def test_inversibility(x_da: xr.DataArray, y_da: xr.DataArray, cfd: dict):
    """Tests whether the reformatting works.

    Args:
        x_da (xr.DataArray): x_inputs
        y_da (xr.DataArray): y_labels
        cfd (dict): dictionary containing configuration
    """
    # run-20210225_161033-yxyit68w/
    x_all, y_all = return_xy_npa(
        x_da, y_da, year=range(cfd["start_year_i"], cfd["end_year_i"])
    )  # all data as numpy.
    x_rp, y_rp = return_xy_npa(
        x_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"])),
        y_npa_to_xr(
            y_all, y_da.isel(year=range(cfd["start_year_i"], cfd["end_year_i"]))
        ),
        year=range(cfd["start_year_i"], cfd["end_year_i"]),
    )
    assert np.all(y_all == y_rp)
