"""Utils for working with vector data in python using shapely and geopandas."""

import collections.abc
import os
import pathlib
from typing import Any, Dict, Iterable, Optional, Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry


def shapely_to_frame(
    shapes: Union[BaseGeometry, Iterable[BaseGeometry]],
    attributes: Optional[Dict[str, Any]] = None,
    crs: Optional[str] = None,
    **kwargs
) -> gpd.GeoDataFrame:
    """
    Turn shapely object into geopandas dataframe.

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


def convert_to_gpkg(
    vector_path: Union[str, os.PathLike],
    save_path: Union[str, os.PathLike],
    driver: str = None,
    rename_class_label: bool = False,
    class_label_column: str = None,
) -> gpd.GeoDataFrame:
    """
    Convert saved vector data file to GPKG format.

    This function allows for the renaming of a column in the loaded dataframe
    to "class_label", which is required to load the data as a `GeoGraph`. The
    `driver` argument also allows for the loading of vector data in any format
    supported by Fiona. A list of available drivers can be found at
    https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py.

    Args:
        vector_path (Union[str, os.PathLike]): A path to a file of vector data.
        save_path (Union[str, os.PathLike]): A path to a GPKG file to save the
        data.
        driver (str, optional): A format to load the data in. Defaults to None,
        which will load a Shapefile.
        rename_class_label (bool, optional): Whether or not to rename a class
        label column. Defaults to False.
        class_label_column (str, optional): The name of the class label column
        to rename. Defaults to None.

    Raises:
        ValueError: If the `save_path` is not a GPKG file.

    Returns:
        gpd.GeoDataFrame: The loaded dataframe.
    """
    vector_path, save_path = pathlib.Path(vector_path), pathlib.Path(save_path)
    if save_path.suffix != ".gpkg":
        raise ValueError("`save_path` must be a GPKG file.")

    drivers = ["ESRI Shapefile"]
    if driver is not None:
        drivers.append(driver)

    df = gpd.read_file(vector_path, enabled_drivers=drivers)
    if rename_class_label:
        df = df.rename(columns={class_label_column: "class_label"})

    df.to_file(save_path, driver="GPKG")
    return df
