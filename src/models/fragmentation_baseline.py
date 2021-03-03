"""Wrappers around pylandstats to automatically compute zonal time-series fragmentation
metrics for a given region from landcover data and specified regions of interest."""
import logging
import os
import pathlib
from copy import copy
from datetime import datetime
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pylandstats as pls
import pyproj
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm

# Initialise the logging:
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _analyse_fragmentation(
    landcover: Union[os.PathLike, xr.DataArray],
    rois: Optional[gpd.GeoDataFrame] = None,
    target_crs: Optional[Union[str, pyproj.CRS]] = None,
    target_x_res: float = 300,
    target_y_res: float = 300,
    no_data: int = 0,
    rois_index_col: str = "name",
    **kwargs
) -> pd.DataFrame:
    """
    Compute pylandstats class fragmentation metrics for ROIs on a landcover map.

    For a list of all computable metrics, see:
    https://pylandstats.readthedocs.io/en/latest/landscape.html

    Args:
        landcover (Union[os.PathLike, xr.DataArray]): The landcover data to use.
        rois (Optional[gpd.GeoDataFrame], optional): A geopandas dataframe which
            contains the list of the geometries for which a class fragmentation analysis
            should be performed. Defaults to None.
        target_crs (Optional[Union[str, pyproj.CRS]], optional): The coordinate
            reference system to use for the class metric computation. For interpretable
            results a CRS with units of meter (e.g. UTM) should be used.
            Defaults to None.
        target_x_res (float, optional): The target pixel resolution along the
            x-direction in the target coordinate reference system. If the CRS has units
            of meter, this corresponds to meters per pixel. Up/downsampling is
            performed via nearest-neighbor sampling with rasterio. Defaults to 300.
        target_y_res (float, optional): The target pixel resolution along the
            y-direction in the target coordinate reference system. If the CRS has units
            of meter, this corresponds to meters per pixel. Up/downsampling is
            performed via nearest-neighbor sampling with rasterio. Defaults to 300.
        no_data (int, optional): The no-data value for the landcover data.
            Defaults to 0.
        rois_index_col (str, optional): Name of the attribute that will distinguish
            region of interest in `rois`. Defaults to "name".
        **kwargs: Keyword arguments of the `compute_class_metrics_df` of pylandstats

    Returns:
        pd.DataFrame: The pandas dataframe containing the computed metrics for each
            landcover class in `landcover` and region of interest given in `rois`
    """

    # 1 Load the data
    if isinstance(landcover, os.PathLike):
        data_original = rxr.open_rasterio(pathlib.Path(landcover))
    elif isinstance(landcover, xr.DataArray):
        data_original = copy(landcover)

    # 2 Reproject to relevant CRS and resolution
    data_reprojected = data_original.rio.reproject(
        target_crs, resolution=(target_x_res, target_y_res)
    )

    # 3 Calculate final resolution
    x_data, y_data = (data_reprojected.x.data, data_reprojected.y.data)
    x_res = abs(x_data[-1] - x_data[0]) / len(x_data)
    y_res = abs(y_data[-1] - y_data[0]) / len(y_data)
    # Free up memory
    del data_original

    # 4 Perform pylandstats analysis on clipped, reprojected region
    # Convert to pylandstats landscape
    data_landscape = pls.Landscape(
        data_reprojected.data.squeeze(),
        res=(x_res, y_res),
        nodata=no_data,
        transform=data_reprojected.rio.transform(),
    )

    # Perform zonal analysis of the rois
    if rois is None:
        zonal_analyser = data_landscape
    else:
        zonal_analyser = pls.ZonalAnalysis(
            data_landscape,
            landscape_crs=target_crs,
            masks=rois,
            masks_index_col=rois_index_col,
        )
    return zonal_analyser.compute_class_metrics_df(**kwargs)


def _fragmentation_dataframes_to_xarray(
    fragmentation_statistics: Dict[Union[str, int, datetime], pd.DataFrame],
    landcover_classes: List[int],
    rois: gpd.GeoDataFrame,
    rois_index_col: str,
) -> xr.DataArray:
    """
    Convert a dictionary of dates and class fragmentation statistics to xarray

    Convenience function to combine temporal, landcover-class, metric and zonal data
    in one datastructure by bunching them together into a multidimensional array.

    Args:
        fragmentation_statistics (Dict[Union[str, int, datetime], pd.DataFrame]): The
            dictionary of fragmentation class statistics. Key corresponds to the date,
            value corresponds to a pandas dataframe of fragmentation metrics as
            computed by pylandstats.compute_class_metrics_df
        landcover_classes (List[int]): A list of all possible landcover classes. Must
            be given as integers.
        rois (gpd.GeoDataFrame): The regions of interest that were analysed in
            `fragmentation_statistics`.
        rois_index_col (str): Name of the attribute that will distinguish region of
            interest in `rois`.

    Returns:
        xr.DataArray: A multi-dimensional array with axes (zones, landcover classes,
            metrics, dates) which contains all fragmentation statistics. Values for
            classes that were not present in the data are `np.nan`
    """

    # 1 Set up axes
    zones = rois[rois_index_col].tolist() if rois is not None else ["all"]
    dates = sorted(fragmentation_statistics.keys())
    metrics = fragmentation_statistics[dates[0]].columns.tolist()

    # 2 Reformat raw fragmentation data into numpy array for transfer to xarray
    raw_data = np.zeros((len(zones), len(landcover_classes), len(metrics), len(dates)))
    for date_index, date in enumerate(dates):
        result = fragmentation_statistics[date]
        for landcover_index, landcover_class in enumerate(landcover_classes):
            try:
                raw_data[:, landcover_index, :, date_index] = result.loc[
                    landcover_class
                ].values
            except KeyError:  # to treat class values that may not appear in a year
                raw_data[:, landcover_index, :, date_index] = np.nan

    # 3 Wrap numpy array into xarray data array object
    fragmentation_data = xr.DataArray(
        raw_data,
        coords=[zones, landcover_classes, metrics, dates],
        dims=["zone", "landcover_class", "metric", "date"],
    )
    return fragmentation_data


def fragmentation_analysis(
    landcover_data: Dict[Union[int, str, datetime], Union[os.PathLike, xr.DataArray]],
    landcover_classes=List[int],
    rois: Optional[gpd.GeoDataFrame] = None,
    target_crs: Optional[Union[str, pyproj.CRS]] = None,
    target_x_res: Optional[float] = 300.0,
    target_y_res: Optional[float] = 300.0,
    no_data: int = 0,
    rois_index_col: str = "name",
    save_path: Union[str, os.PathLike, None] = None,
) -> xr.DataArray:
    """
    Compute time-series of pylandstas class metrics for given ROIs in landcover data.

    For a list of all computable metrics, see:
    https://pylandstats.readthedocs.io/en/latest/landscape.html

    Args:
        landcover_data (Dict): The landcover data to use. Should be a dictionary with
            dates (str, int, datetime) as keys and paths to landcover raster data or
            landcover data arrays as values.
        landcover_classes(List[int]): A list of all possible landcover classes
        rois (Optional[gpd.GeoDataFrame], optional): A geopandas dataframe which
            contains the list of the geometries for which a class fragmentation analysis
            should be performed. Defaults to None.
        target_crs (Optional[Union[str, pyproj.CRS]], optional): The coordinate
            reference system to use for the class metric computation. For interpretable
            results a CRS with units of meter (e.g. UTM) should be used.
            Defaults to None.
        target_x_res (float, optional): The target pixel resolution along the
            x-direction in the target coordinate reference system. If the CRS has units
            of meter, this corresponds to meters per pixel. Up/downsampling is
            performed via nearest-neighbor sampling with rasterio. Defaults to 300.
        target_y_res (float, optional): The target pixel resolution along the
            y-direction in the target coordinate reference system. If the CRS has units
            of meter, this corresponds to meters per pixel. Up/downsampling is
            performed via nearest-neighbor sampling with rasterio. Defaults to 300.
        no_data (int, optional): The no-data value for the landcover data.
            Defaults to 0.
        rois_index_col (str, optional): Name of the attribute that will distinguish
            region of interest in `rois`. Defaults to "name".
        save_path (Union[str, os.PathLike, None]): The path at which to save the outcome
            of the analysis.
        **kwargs: Keyword arguments of the `compute_class_metrics_df` of pylandstats

    Returns:
        xr.DataArray: A multi-dimensional array with axes (zones, landcover classes,
            metrics, dates) which contains all fragmentation statistics. Values for
            classes that were not present in the data are `np.nan`
    """

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        assert save_path.parent.exists(), "Save path directories does not exist."
        assert not save_path.exists(), "File already exists. Override not supported."
    assert rois is None or isinstance(rois, gpd.GeoDataFrame)

    fragmentation_statistics = {}
    for date, landcover in tqdm(landcover_data.items()):
        logging.info("Analysing data for %s", date)
        fragmentation_statistics[date] = _analyse_fragmentation(
            landcover=landcover,
            rois=rois,
            target_crs=target_crs,
            target_x_res=target_x_res,
            target_y_res=target_y_res,
            no_data=no_data,
            rois_index_col=rois_index_col,
        )

    # Convert final data to xarray
    logging.info("Analysis completed. Converting data to xarray.")
    fragmentation_data = _fragmentation_dataframes_to_xarray(
        fragmentation_statistics=fragmentation_statistics,
        landcover_classes=landcover_classes,
        rois=rois,
        rois_index_col=rois_index_col,
    )

    # Save data to disk for further analysis
    if save_path is not None:
        fragmentation_data.to_netcdf(save_path)
        save_path.chmod(0o664)
        logging.info("Saved data at %s.", save_path)

    return fragmentation_data
