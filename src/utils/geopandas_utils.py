"""Helper functions for operating with geopandas objects."""
from typing import Dict, List

import geopandas as gpd
import networkx as nx
from shapely.geometry import MultiPolygon

from src.utils.polygon_utils import (
    connect_with_interior_bulk,
    connect_with_interior_or_edge_bulk,
    connect_with_interior_or_edge_or_corner_bulk,
)

# For switching identifiction mode in `identify_node`
_BULK_SPATIAL_IDENTIFICATION_FUNCTION = {
    "corner": connect_with_interior_or_edge_or_corner_bulk,
    "edge": connect_with_interior_or_edge_bulk,
    "interior": connect_with_interior_bulk,
}


def identify_node(
    node: dict, other_df: gpd.GeoDataFrame, mode: str = "corner"
) -> List[int]:
    """
    Return list of all `loc` in `other_df` which identify with the given `node`.

    Args:
        node (dict): The node for which to find nodes in `other_df` that can be
            identified with `node`.
        other_df (GeoDataFrame): The GeoDataFrame object in which to search for
            identifications
        mode (str, optional): Must be one of `corner`, `edge` or `interior`. Defaults
            to "corner".
            The different modes correspond to different rules for identification:

            - corner: Polygons of the same `class_label` which overlap, touch in their
              edges or corners will be identified with each other. (fastest)
            - edge: Polygons of the same `class_label` which overlap or touch in their
              edges will be identified with each other.
            - interior: Polygons of the same `class_label` which overlap will be
              identified with each other. Touching corners or edges are not counted.

    Returns:
        np.ndarray: List of node `loc` in `other_df` which identify with `node`.
    """
    # Mode switch
    assert mode in ["corner", "edge", "interior"]
    have_valid_overlap = _BULK_SPATIAL_IDENTIFICATION_FUNCTION[mode]

    # Get potential candidates for overlap
    candidate_ids = other_df.sindex.query(node["geometry"], sort=True)
    # Filter candidates according to the same class label
    candidate_ids = candidate_ids[
        other_df["class_label"].values[candidate_ids] == node["class_label"]
    ]
    # Filter candidates accroding to correct spatial overlap
    candidate_ids = candidate_ids[
        have_valid_overlap(node["geometry"], other_df.geometry.values[candidate_ids])
    ]

    return other_df.index.values[candidate_ids].tolist()


def identify_dfs(
    df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame, mode: str
) -> Dict[int, List[int]]:
    """
    Idenitfy all nodes from `graph1` with nodes from `graph2` based on the given `mode`

    Args:
        df1 (GeoDataFrame): The dataframe whose node indicies will form the domain
        df2 (GeoDataFrame): The dataframe whose node indices will form the
            image (target)
        mode (str): The mode to use for node identification. Must be one of `corner`,
            `edge` or `interior`.
            The different modes correspond to different rules for identification:

            - corner: Polygons of the same `class_label` which overlap, touch in their
              edges or corners will be identified with each other. (fastest)
            - edge: Polygons of the same `class_label` which overlap or touch in their
              edges will be identified with each other.
            - interior: Polygons of the same `class_label` which overlap will be
              identified with each other. Touching corners or edges are not counted.

    Returns:
        mapping (Dict[int, np.ndarray]): A dictionary that represents the map from
            elements of `df1` to `df2`.
    """

    assert df1.crs == df2.crs, "CRS systems do not agree."
    mapping = {index1: [] for index1 in df1.index}

    for index in df1.index:  # TODO: Speed up & enable trivial parallelisation
        mapping[index] = identify_node(df1.loc[index], df2, mode=mode)

    return mapping


def merge_diagonally_connected_polygons(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return a new dataframe with all geometries of `df` which touch at corners merged.

    Merged geometries will be of type shapely.geometry.MultiPolygon

    Args:
        df (gpd.GeoDataFrame): The dataframe to analyse for geometries which touch
            at corners

    Returns:
        gpd.GeoDataFrame: The dataframe with patches that touch at corners merged
    """

    # Identify the nodes that will be merged
    mapping = identify_dfs(df, df, mode="corner")
    mapping_graph = nx.from_dict_of_lists(mapping)
    nodes_to_merge = [
        list(group)
        for group in nx.algorithms.connected_components(mapping_graph)
        if len(group) > 1
    ]

    # Remove nodes that will be merged
    nodes_to_merge_flattened = [item for sublist in nodes_to_merge for item in sublist]
    new_df = df.drop(nodes_to_merge_flattened)

    # Add the new, merged nodes and reset the index
    new_nodes = {"geometry": [], "class_label": []}
    for nodes in nodes_to_merge:
        new_nodes["class_label"].append(df["class_label"].loc[nodes[0]])
        new_nodes["geometry"].append(MultiPolygon(df["geometry"].loc[nodes].values))

    return new_df.append(gpd.GeoDataFrame(new_nodes), ignore_index=True)
