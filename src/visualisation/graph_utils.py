"""This module contains utility function for generally plotting graphs."""

from __future__ import annotations

from typing import Tuple

import geopandas as gpd
import shapely.geometry

from src.constants import WGS84
import src.geograph


def create_node_edge_geometries(
    graph: src.geograph.GeoGraph,
    crs: str = WGS84,
) -> Tuple[gpd.GeoSeries, gpd.GeoSeries]:
    """Create node and edge geometries for the networkx graph G.

    Returns node and edge geometries in two GeoDataFrames. The output can be used for
    plotting a graph.

    Args:
        graph (GeoGraph): graph with nodes and edges
        crs (str, optional): coordinate reference system of graph. Defaults to UTM35N.

    Returns:
        Tuple[gpd.GeoSeries, gpd.GeoSeries]: dataframes of nodes and edges
            respectively.
    """

    node_geoms = graph.df.geometry.to_crs(crs).representative_point()
    rep_points = node_geoms.to_dict()

    edge_lines = {}
    for idx, (node_a, node_b) in enumerate(graph.graph.edges()):
        point_a = rep_points[node_a]
        point_b = rep_points[node_b]
        edge_lines[idx] = shapely.geometry.LineString([point_a, point_b])
    edge_geoms = gpd.GeoSeries(edge_lines)

    return node_geoms, edge_geoms
