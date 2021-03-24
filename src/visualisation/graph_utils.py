"""This module contains utility function for generally plotting graphs."""

from __future__ import annotations

from typing import Tuple

import geopandas as gpd
import shapely.geometry

from src.constants import UTM35N
import src.geograph


def create_node_edge_geometries(
    graph: src.geograph.GeoGraph,
    crs: str = UTM35N,
    include_edges: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create node and edge geometries for the networkx graph G.

    Returns node and edge geometries in two GeoDataFrames. The output can be used for
    plotting a graph.

    Args:
        graph (nx.Graph): graph with nodes and edges
        crs (str, optional): coordinate reference system of graph. Defaults to UTM35N.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: dataframes of nodes and edges
            respectively.
    """

    node_geoms = graph.df.representative_point()
    rep_points = node_geoms.to_dict()

    if include_edges:
        edge_lines = {}
        for idx, (node_a, node_b) in enumerate(graph.graph.edges()):
            point_a = rep_points[node_a]
            point_b = rep_points[node_b]
            edge_lines[idx] = shapely.geometry.LineString([point_a, point_b])
        edge_geoms = gpd.GeoSeries(edge_lines)
        edge_geoms.set_crs(crs)
    else:
        edge_geoms = None

    return node_geoms, edge_geoms
