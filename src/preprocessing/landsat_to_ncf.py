"""
src/preprocessing/landsat.py
================================================================
from src.preprocessing.landsat import return_path_dataarray

https://www.usgs.gov/faqs/what-are-best-landsat-spectral-bands-use-my-research?qt-news_science_products=0#qt-news_science_products

================================================================
Landsat 8 Operational Land Image (OLI) and Thermal Infrared Sensor (TIRS)

Band |	Wavelength | Useful for mapping
Band 1 - coastal aerosol |	0.43-0.45 |	Coastal and aerosol studies
Band 2 - blue	| 0.45-0.51 |	Bathymetric mapping, distinguishing soil from vegetation and deciduous from coniferous vegetation
Band 3 - green |	0.53-0.59 |	Emphasizes peak vegetation, which is useful for assessing plant vigor
Band 4 - red | 	0.64-0.67  | Discriminates vegetation slopes
Band 5 - Near Infrared (NIR) |	0.85-0.88 | Emphasizes biomass content and shorelines
Band 6 - Short-wave Infrared (SWIR) 1	| 1.57-1.65 | 	Discriminates moisture content of soil and vegetation; penetrates thin clouds
Band 7 - Short-wave Infrared (SWIR) 2  |  2.11-2.29	| Improved moisture content of soil and vegetation; penetrates thin clouds

================================================================
Landsat 4-5 Thematic Mapper (TM) and Landsat 7 Enhanced Thematic Mapper Plus (ETM+)
Band	Wavelength	Useful for mapping
Band 1 - blue	0.45-0.52	Bathymetric mapping, distinguishing soil from vegetation and deciduous from coniferous vegetation
Band 2 - green	0.52-0.60	Emphasizes peak vegetation, which is useful for assessing plant vigor
Band 3 - red	0.63-0.69	Discriminates vegetation slopes
Band 4 - Near Infrared	0.77-0.90	Emphasizes biomass content and shorelines
Band 5 - Short-wave Infrared	1.55-1.75	Discriminates moisture content of soil and vegetation; penetrates thin clouds
Band 6 - Thermal Infrared	10.40-12.50	Thermal mapping and estimated soil moisture
Band 7 - Short-wave Infrared	2.09-2.35	Hydrothermally altered rocks associated with mineral deposits


==================================
Vis Bands:

yr 1
('B3', 'B2', 'B1')

yr 39
('B1', 'B4', 'B3', 'B2')

# L5
'B3', 'B2', 'B1', 'B4', 'B5', 'B7'
# L8
'B4', 'B3', 'B2', 'B5', 'B6', 'B7'

IR Bands:
chern  AMJ  20/37 
('B4', 'B5', 'B7') - near ir, short ir A, short ir B
chern  AMJ 30/37 
('B5', 'B6', 'B7') - near ir, SWIR 1, SWIR 2


==================================
More information on the bands:
http://web.pdx.edu/~nauna/resources/10_BandCombinations.htm


"""
import os
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import dask
import xarray as xr
import rasterio
from src.utils import timeit
from src.constants import SAT_DIR


@timeit
def return_path_dataarray():
    """
    makes path to the google earth engine landsat data.
    if the file doesn't exist, it becomes 'None' in the netcdf.
    :return: xarray.DataArray containing paths to Landsat data.
    """
    incomplete_years = []  # [1984, 1994, 2002, 2003, 2008]
    # previously these were ignored, now algorithm is robust to absence.
    years = [year for year in range(1984, 2021) if year not in incomplete_years]
    im_type = ["hab", "chern"]
    month_groups = ["JFM", "AMJ", "JAS", "OND"]
    ir = [0, 1]
    directory = SAT_DIR
    # year, month_group, im_type
    path_array = np.empty(
        [len(years), len(month_groups), len(im_type), len(ir)], dtype=object
    )
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
                if "IR" in full_name:
                    path_array[indices[0], indices[1], indices[2], 1] = full_name
                else:
                    path_array[indices[0], indices[1], indices[2], 0] = full_name
    return xr.DataArray(
        data=path_array,
        dims=["year", "mn", "ty", "ir"],
        coords=dict(
            year=years,
            mn=month_groups,
            ty=im_type,
            ir=ir,
        ),
        attrs=dict(
            description="Paths to tif.",
        ),
    )


@timeit
def return_normalized_array(
    one,
    two,
    three,
    filter_together=True,
    high_limit=1.5e3,
    low_limit=0,
    high_filter=True,
    low_filter=True,
    common_norm=True,
):
    """
    Function takes numpy bands and converts it to a preprocessed numpy array.
    :param filter_together: if True will only display points where all 3 members of a band
    are below the threshold.
    :param high_limit: The aforementioned threshold.
    :param low_limit: Adding a lower threshold.
    :param high_filter: Bool, whether to turn the high limit on.
    :param low_filter: Bool, whether to turn the lower limit on.
    :param common_norm: Bool, whether to norm between the upper and lower limit.
    :return: numpy float array
    """
    print('high_limit \t', high_limit)

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

    def comb_and_filt(red, green, blue, filter_red, filter_green, filter_blue):
        filter_array = np.logical_or(
            np.logical_or(filter_red, filter_green), filter_blue
        )
        return (
            filt(red, filter_array),
            filt(green, filter_array),
            filt(blue, filter_array),
        )

    def filter_sep_and_norm(array):
        if high_filter:
            array = filt(array, array >= high_limit).filled(np.nan)
        if low_filter:
            array = filt(array, array <= low_limit).filled(np.nan)
        return norm(array)

    def filter_tog_and_norm(red, green, blue):
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
        one_norm, two_norm, three_norm = (
            filter_sep_and_norm(one),
            filter_sep_and_norm(two),
            filter_sep_and_norm(three),
        )
    else:
        one_norm, two_norm, three_norm = filter_tog_and_norm(one, two, three)
    # Stack bands
    return np.dstack((one_norm, two_norm, three_norm))


def load_rgb_data(
    file_name=os.path.join(SAT_DIR, "2012/L7_chern_2012_AMJ.tif"), high_limit=1500,
):
    """
    :param file_name: full path to .tif image.
    """
    # Open the file:
    raster = rasterio.open(file_name)
    # Convert to numpy arrays
    if "L8" in file_name and "IR" not in file_name:
        # LandSat8 currently stored in different bands for R, G, B.
        ins = [2, 3, 4]
    else:
        ins = [1, 2, 3]

    one, two, three = raster.read(ins[0]), raster.read(ins[1]), raster.read(ins[2])
    print(raster.descriptions)
    descriptions = [
        raster.descriptions[ins[0] - 1],
        raster.descriptions[ins[1] - 1],
        raster.descriptions[ins[2] - 1],
    ]

    """
    if "IR" not in file_name:
        # for color arrays the order needs to be rgb not bgr
        descriptions.reverse()
        return return_normalized_array(three, two, one, **kwargs), descriptions
    else:
        return return_normalized_array(one, two, three, **kwargs), descriptions
    """
    return return_normalized_array(one, two, three, high_limit=high_limit), descriptions


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
    ir_name = ["", "_IR"]
    high_limits = [1500, 4000]
    path_da = return_path_dataarray()
    for ty, ty_v in [
        (1, "chern")
    ]:  # enumerate(path_da.coords["ty"].values.tolist()):  # [(1, "chern")]:
        for mn, mn_v in enumerate(path_da.coords["mn"].values.tolist()): 
            """
                         #[(0, "JFM"),
                         #(1, "AMJ")
                         #(2, "JAS"),
                         #(3, "OND")
                         #]:
            """
            #  [(1, "AMJ"), (2, "JAS"), (3, "OND")]:  # enumerate(path_da.coords["mn"].values.tolist()):
            for ir in [0]: #[1]:  #[0, 1]
                path_list = []
                for year in tqdm(
                    range(len(path_da.coords["year"].values)),
                    ascii=True,
                    desc=ty_v + "  " + mn_v,
                ):
                    file_name = path_da.isel(
                        year=year, mn=mn, ty=ty, ir=ir
                    ).values.tolist()
                    print(file_name)
                    tmp_name = os.path.join(
                        tmp_path,
                        ty_v
                        + "_"
                        + mn_v
                        + "_year_"
                        + str(path_da.isel(year=year).coords["year"].values)
                        + ir_name[ir]
                        + ".nc",
                    )
                    if file_name != None and os.path.exists(file_name):
                        xr_da = xr.open_rasterio(file_name)
                        data, descriptions = load_rgb_data(file_name, high_limit=high_limits[ir])
                    else:
                        if ty_v == "chern" and ir == 0:
                            file_name = os.path.join(
                                SAT_DIR, "2012/L7_chern_2012_AMJ.tif"
                            )
                        if ty_v == "chern" and ir == 1:
                            file_name = os.path.join(
                                SAT_DIR, "2012/L7_chern_2012_AMJ_IR.tif"
                            )
                        elif ty_v == "hab" and ir == 0:
                            file_name = os.path.join(
                                SAT_DIR, "2012/L7_hab_2012_AMJ.tif"
                            )
                        elif ty_v == "hab" and ir == 1:
                            file_name = os.path.join(
                                SAT_DIR, "2012/L7_hab_2012_AMJ_IR.tif"
                            )
                        xr_da = xr.open_rasterio(file_name)
                        data, descriptions = load_rgb_data(file_name, high_limit=high_limits[ir])
                        data[
                            :
                        ] = np.nan  # make everything nan if the file didn't exist.
                    if ir == 0:
                        band_names = ["red", "green", "blue"]
                    else:
                        band_names = ["nir", "swir1", "swir2"]
                    xr.DataArray(
                        data=np.expand_dims(np.expand_dims(data, axis=3), axis=4),
                        dims=["y", "x", "band", "year", "mn"],
                        coords=dict(
                            y=xr_da.coords["y"].values,
                            x=xr_da.coords["x"].values,
                            band=band_names,
                            year=[path_da.isel(year=year).coords["year"].values],
                            mn=[mn_v],
                        ),
                        attrs=dict(
                            description=(
                                "Normalized reflectance at "
                                + ty_v
                                + " for "
                                + mn_v
                                + ". Bands order "
                                + str(descriptions)
                                + "."
                            ),
                            bands=descriptions,
                        ),
                    ).astype("float32").to_dataset(name="norm_refl").to_netcdf(tmp_name)
                    del data
                    del xr_da
                    path_list.append(tmp_name)
                # this option should override chunk size to prevent chunks from being too large.
                with dask.config.set(**{"array.slicing.split_large_chunks": True}):
                    # https://github.com/pydata/xarray/issues/3961

                    @timeit
                    def _cat_ds():
                        print("paths for concat da", path_list)
                        # return  xr.concat(da_list, "year")
                        return xr.open_mfdataset(
                            path_list,
                            concat_dim="year",
                            chunks={"year": 1}, # {"band": 1, "year": 1},  # parallel=True,
                        )

                    cat_ds = _cat_ds()
                    directory = os.path.join(SAT_DIR, "nc_" + ty_v)
                    if not os.path.exists(directory):
                        os.mkdir(directory)

                    @timeit
                    def _save_nc(cat_ds):
                        name = os.path.join(
                            directory, ty_v + "_" + mn_v + ir_name[ir] + ".nc"
                        )
                        print("about to save " + name)
                        cat_ds.load().to_netcdf(name)
                        print("finished saving")

                    _save_nc(cat_ds)

                    del cat_ds
