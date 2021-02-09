"""Utils for working with vector data in python using shapely and geopandas"""

import geopandas as gpd
import collections.abc
from shapely.geometry.base import BaseGeometry
from typing import Union, Iterable, Dict, Any, Optional


def shapely_to_frame(
    shapes: Union[BaseGeometry, Iterable[BaseGeometry]],
    attributes: Optional[Dict[str, Any]] = None,
    crs: Optional[str] = None,
    **kwargs
) -> gpd.GeoDataFrame:
    """
    Turn shapely object into geopandas dataframe

    Convenience function to quickly turn shapely objects to geopandas dataframes
    within one line, regardless of whether they are a single object (e.g. Point)
    or a collection of multiple object (e.g. Point Collection).

    Args:
        shapes (Union[BaseGeometry, Iterable[BaseGeometry]]): The shape or collection
            of shapes to turn into a geopandas dataframe.
        attributes (Optional[Dict[str, Any]], optional): The attributes for each item of
            `shapes` to add to the geopandas dataframe. Dictionary keys will be used as
            column names. Dictionary values will be used as columns (i.e. must have same
            length as `shapes`). Defaults to None.
        crs (Optional[str], optional): The coordinate reference to use. Should be given
            in crs string from (e.g. `EPSG:4326`). Defaults to None.
        **kwargs: Columns can also be passed as kwargs instead of via the attributes
            dict.

    Returns:
        gpd.GeoDataFrame: The geodataframe that contains the shapes and their attributes
    """

    is_multiple = isinstance(shapes, collections.abc.Iterable)

    # Add shapely geometries
    data = {"geometry": shapes if is_multiple else [shapes]}
    # Add attributes as columns
    if attributes is not None:
        for key, val in attributes.items():
            data[key] = val if is_multiple else [val]
    # Add any passed kwargs as columns
    for key, val in kwargs.items():
        data[key] = val if is_multiple else [val]
    # Convert to dataframe and return
    return gpd.GeoDataFrame(data, crs=crs)
