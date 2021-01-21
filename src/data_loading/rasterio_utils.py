"""A collection of utility functions for data loading with rasterio"""

from typing import Optional, Union, Iterable, Tuple
import numpy as np
from rasterio.io import DatasetReader
from rasterio.crs import CRS


class CoordinateSystemError(Exception):
    """Basic exception for coordinate system errors"""


class InvalidUseError(Exception):
    """Basic exception for invalid usage of functions"""


def get_thumbnail(
    data: DatasetReader,
    band_idx: Optional[int] = 1,
    height: Optional[int] = 100,
    width: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate a thumbnail for a given band of a rasterio data

    Args:
        data (DatasetReader): rasterio data handle
        band_idx (Optional[int], optional): The band index for which to caluclate the
            thumbnail. Defaults to 1.
        height (Optional[int], optional): The desired height of the thumbnail.
            If only the height is set, the width will be automatically determined
            from the datas aspect ratio. Defaults to 100.
        width (Optional[int], optional): The desired width of the thumbnail.
            Defaults to None.

    Returns:
        np.ndarray: The 2D numpy array representing the thumbnail as calculated
            from nearest neighbour resampling.
    """

    aspect_ratio = data.height / data.width

    if height and not width:
        width = int(height / aspect_ratio)
    elif width and not height:
        height = int(width * aspect_ratio)

    assert height > 0 and width > 0, "Output height and/or width must be specified."

    return data.read(band_idx, out_shape=(int(height), int(width)))


def read_from_lat_lon(
    data: DatasetReader,
    band_idxs: Union[int, Iterable[int]],
    lat: Tuple[float],
    lon: Tuple[float],
    **kwargs,
) -> np.ndarray:
    """
    Read in a tile of raster data form specified latitude and longitude values.

    Note: This function only works if `data` is provided in the WGS geographical
        coordinate system (Note: WGS84 = EPSG4326).

    Args:
        data (DatasetReader): rasterio data handle
        band_idxs (Union[int, Iterable[int]]): The band index or indices for which to
            read the information from the underlying rasterio `data`.
        lat (Tuple[float]): A tuple containing (latitude_min, latitude_max).
            Latitudes must be in the range (-90, 90).
        lon (Tuple[float]): A tuple containing (longitude_min, longitude_max).
            Longitudes must be in the range (-180, 180).

    Returns:
        np.ndarray: A multidimensional numpy array containing the specified bands in
            the given latitude, longitude bounds.
    """
    # Check that geographical cooridnate reference system of the data is
    #  WGS84 (i.e. epsg 4326). If this is not the case, prompt user to retransform.
    if not data.crs == CRS.from_epsg(4326):
        raise CoordinateSystemError(
            "Latitude, Longitude based reading requires EPSG 4326 (WGS84) coordinate "
            f"reference system. Current CRS is {data.crs}. To use this method, first "
            "transform your data to EPSG 4326."
        )

    lat_min, lat_max = lat
    lon_min, lon_max = lon

    # Check if latitude and longitude are within physical bounds
    if not -90 <= lat_min < lat_max <= 90:
        raise InvalidUseError(
            "Latitudinal coordinates must be between -90 and 90 degrees"
        )
    if not -180 <= lon_min < lon_max <= 180:
        raise InvalidUseError(
            "Longitudinal coordinates must be between -180 and 180 degrees"
        )

    # Note: rows increase from top to bottom, latitude increases from bottom to top
    row_min, col_min = data.index(lon_min, lat_max)  # index swaps values internally
    row_max, col_max = data.index(lon_max, lat_min)

    # Read specified region
    return data.read(
        indexes=band_idxs, window=((row_min, row_max), (col_min, col_max)), **kwargs
    )
