"""Contains tools for binary operations between GeoGraph objects."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

import src.utils.geopandas_utils as gpd_utils
from src.utils.polygon_utils import EMPTY_POLYGON, collapse_empty_polygon

if TYPE_CHECKING:
    from src import geograph


class NodeMap:
    """Class to store node mappings between two graphs (the src_graph and trg_graph)"""

    def __init__(
        self,
        src_graph: geograph.GeoGraph,
        trg_graph: geograph.GeoGraph,
        mapping: Dict[int, List[int]],
    ) -> None:
        """
        Class to store node mappings between two graphs (`trg_graph` and `src_graph`)

        This class stores a dictionary of node one-to-many relationships of nodes from
        `src_graph` to `trg_graph`. It also provides support for convenient methods for
        inverting the mapping and bundles the mapping information with references to
        the `src_graph` and `trg_graph`

        Args:
            src_graph (GeoGraph): Domain of the node map (keys in `mapping` correspond
                to indices from the `src_graph`).
            trg_graph (GeoGraph): Image of the node map (values in `mapping` correspond
                to indices from the `trg_graph`)
            mapping (Dict[int, List[int]], optional): A lookup table for the map which
                maps nodes form `src_graph` to `trg_graph`.
        """
        self._src_graph = src_graph
        self._trg_graph = trg_graph
        self._mapping = mapping

    @property
    def src_graph(self) -> geograph.GeoGraph:
        """Keys in the mapping dict correspond to node indices in the `src_graph`"""
        return self._src_graph

    @property
    def trg_graph(self) -> geograph.GeoGraph:
        """Values in the mapping dict correspond to node indices in the `trg_graph`"""
        return self._trg_graph

    @property
    def mapping(self) -> Dict[int, List[int]]:
        """
        Look-up table connecting node indices from `src_graph` to those of `trg_graph`.
        """
        return self._mapping

    def __invert__(self) -> NodeMap:
        """Compute the inverse NodeMap"""
        return self.invert()

    def __eq__(self, other: NodeMap) -> bool:
        """Check two NodeMaps for equality"""
        return (
            self.src_graph == other.src_graph
            and self.trg_graph == other.trg_graph
            and self.mapping == other.mapping
        )

    def invert(self) -> NodeMap:
        """Compute the inverse NodeMap from `trg_graph` to `src_graph`"""
        inverted_mapping = {index: [] for index in self.trg_graph.df.index}

        for src_node in self.src_graph.df.index:
            for trg_node in self.mapping[src_node]:
                inverted_mapping[trg_node].append(src_node)

        return NodeMap(
            src_graph=self.trg_graph, trg_graph=self.src_graph, mapping=inverted_mapping
        )


def identify_node(
    node: dict, other_graph: geograph.GeoGraph, mode: str = "corner"
) -> List[int]:
    """
    Return list of all node ids in `other_graph` which identify with the given `node`.

    Args:
        node (dict): The node for which to find nodes in `other_graphs` that can be
            identified with `node`.
        other_graph (GeoGraph): The GeoGraph object in which to search for
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
        List[int]: List of node ids in `other_graph` which identify with `node`.
    """
    return gpd_utils.identify_node(node, other_graph.df, mode=mode)


def identify_graphs(
    graph1: geograph.GeoGraph, graph2: geograph.GeoGraph, mode: str
) -> NodeMap:
    """
    Idenitfy all nodes from `graph1` with nodes from `graph2` based on the given `mode`

    Args:
        graph1 (GeoGraph): The GeoGraph whose node indicies will form the domain
        graph2 (GeoGraph): The GeoGraph whose node indices will form the image (target)
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
        NodeMap: A NodeMap containing the map from `graph1` to `graph2`.
    """
    mapping = gpd_utils.identify_dfs(graph1.df, graph2.df, mode=mode)

    return NodeMap(src_graph=graph1, trg_graph=graph2, mapping=mapping)


def graph_polygon_diff(node_map: NodeMap) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Return the (multi)polygon areas that were added/removed when going
    from `src_graph` to `trg_graph`.

    Args:
        node_map (NodeMap): The node map from `src_graph` to `trg_graph`

    Returns:
        Tuple[GeoDataFrame, GeoDataFrame]: Added parts and removed parts as geopandas
            GeoDataFrame objects with the same index and crs as the src graph.
    """
    assert (
        node_map.src_graph.crs == node_map.trg_graph.crs
    ), "CRS systems of graphs do not agree."

    trg_minus_src = []
    src_minus_trg = []
    for index in node_map.src_graph.df.index:
        added_part, removed_part = node_polygon_diff(index, node_map)
        trg_minus_src.append(added_part)
        src_minus_trg.append(removed_part)

    trg_minus_src = gpd.GeoDataFrame(
        index=node_map.src_graph.df.index,
        geometry=trg_minus_src,
        crs=node_map.src_graph.crs,
    )

    src_minus_trg = gpd.GeoDataFrame(
        index=node_map.src_graph.df.index,
        geometry=src_minus_trg,
        crs=node_map.src_graph.crs,
    )

    return trg_minus_src, src_minus_trg


def node_polygon_diff(
    src_node_id: int, node_map: NodeMap
) -> Tuple[BaseGeometry, BaseGeometry]:
    """
    Return the (multi)polygon areas that were added/removed from the given node.

    Args:
        src_node_id (int): The id of the node in `src_graph` to check.
        node_map (NodeMap): The node map object between `src_graph` and `trg_graph`

    Returns:
        Tuple[BaseGeometry, BaseGeometry]: Added part and removed part as shapely
            BaseGeometry objects.
    """

    src_polygon: Polygon = node_map.src_graph.df.geometry.loc[src_node_id]
    trg_node_ids: List[int] = node_map.mapping[src_node_id]

    if len(trg_node_ids) > 0:
        trg_polygon: Polygon = node_map.trg_graph.df.geometry.loc[
            trg_node_ids
        ].unary_union
        removed_part = collapse_empty_polygon(src_polygon.difference(trg_polygon))
        added_part = collapse_empty_polygon(trg_polygon.difference(src_polygon))

    else:
        removed_part = src_polygon
        added_part = EMPTY_POLYGON

    return added_part, removed_part
