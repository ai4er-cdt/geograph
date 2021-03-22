"""Convenience functions for creating and analysing test data for GeoGraph"""
from typing import Iterable, Tuple

import affine
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.rasterio_utils import polygonise
from src.geograph.geograph import GeoGraph

# Mirror the x axis
AFFINE_MIRROR_X = affine.Affine(-1, 0, 0, 0, 1, 0)
# Mirror the y axis
AFFINE_MIRROR_Y = affine.Affine(1, 0, 0, 0, -1, 0)


def get_array_transform(arr: np.ndarray, xoff: int = 0, yoff: int = 0) -> affine.Affine:
    """
    Return affine transform for np.array such that lower-left corner conicides with
    (xoff, yoff).

    Note:
        This function is meant for use with `polygonise` to create simple test cases
        of polygon data and position them at the desired offset.

    Args:
        arr (np.ndarray): The numpy array for which the affine transform will be
        calculated
        xoff (int, optional): x-offset (horizontal) of the origin. Defaults to 0.
        yoff (int, optional): y-offset (vertical) of the origin. Defaults to 0.

    Returns:
        affine.Affine: The affine transformation that places the lower-left corner of
            the given numpy array at (xoff, yoff).
    """

    return affine.Affine.translation(xoff, yoff + arr.shape[0]) * AFFINE_MIRROR_Y


def _xy_to_rowcol(
    arr: np.ndarray, x: Tuple[int, int], y: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Transform x-y indexing to row-column indexing for accessing the given numpy array.

    Convenience function to transfrom x-y indices (origin: lower left corner)
    to row-column indices (origin: upper left corner).

    Note:
        x,y indices must be non-negative or `None`.

    Args:
        arr (np.ndarray): The numpy array for which to transform the indices
        x (Tuple[int, int]): The x indices. Must be >= 0 or None.
        y (Tuple[int, int]): The y indices. Must be >= 0 or None.

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: The row and column indicies
            for accessing the numpy array `arr`
    """

    # Throw error if negative inidices are given (positive & None indices are valid)
    is_valid = lambda _val: (_val is None) or _val >= 0
    assert all(map(is_valid, x)) and all(map(is_valid, y)), "invalid xy coordinates"

    # Convert
    row = (arr.shape[0] - (y[1] or arr.shape[0]), arr.shape[0] - (y[0] or 0))
    col = x

    return row, col


def polygonise_sub_array(
    arr: np.ndarray, x_lims: Tuple[int, int], y_lims: Tuple[int, int]
) -> gpd.GeoDataFrame:
    """
    Convert sub-array of a given numpy array into polygons.

    Note:
        x-y indexing is used for convenience with plotting later. The origin for
        x-y indexing is taken to be at the lower left corner of the array. The
        x-index increases horizontally to the right, y-index increases vertically to
        the top.

    Args:
        arr (np.ndarray): The numpy array from which to select the sub-array
        x_lims (Tuple[int, int]): The x-limits of the sub-array. Must be >=0 or None.

        y_lims (Tuple[int, int]): The y-limits of the sub-array. Must be >=0 or None.

    Returns:
        gpd.GeoDataFrame: The polygons created from the numpy array.
    """

    # Convert x-y indexing to row-col indexing
    row_lims, col_lims = _xy_to_rowcol(arr, x_lims, y_lims)
    # Select sub array
    sub_array = arr[row_lims[0] : row_lims[1], col_lims[0] : col_lims[1]]

    return polygonise(
        sub_array,
        transform=get_array_transform(sub_array, xoff=x_lims[0], yoff=y_lims[0]),
    )


def plot_identified_nodes(
    node: dict, other_graph: GeoGraph, identified_nodes: Iterable[int]
) -> None:
    """
    Plot nodes that identify with `node` in `other_graph`

    Args:
        node (dict): The node for which identification checks were performed (will be
            colored with a blue frame)
        other_graph (GeoGraph): The geograph of nodes with which the given node was
            compared
        identified_nodes (Iterable[int]): The list of node ids in `other_graph` with
            which the current `node` was identified
    """

    candidate_ids = list(other_graph.rtree.intersection(node["geometry"].bounds))

    # Create color palette dependent on existing class labels
    class_labels = set(other_graph.df.loc[candidate_ids, "class_label"])
    class_labels.add(node["class_label"])
    colors = sns.color_palette("hls", len(class_labels))
    map_to_color = dict(zip(class_labels, colors))

    xs, ys = node["geometry"].exterior.xy
    plt.fill(xs, ys, alpha=0.4, fc=map_to_color[node["class_label"]], ec=None)
    plt.plot(xs, ys, color="blue", linewidth=6)

    for node_id in candidate_ids:
        other_node = other_graph.df.iloc[node_id]
        xs, ys = other_node["geometry"].exterior.xy
        plt.fill(xs, ys, alpha=0.4, fc=map_to_color[other_node["class_label"]], ec=None)

    for node_id in identified_nodes:
        other_node = other_graph.df.iloc[node_id]
        xs, ys = other_node["geometry"].exterior.xy
        plt.fill(xs, ys, alpha=0.4, fc=map_to_color[other_node["class_label"]], ec=None)
        plt.plot(xs, ys, color="green", linewidth=3)

    plt.show()
