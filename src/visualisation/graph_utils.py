"""This module contains utility function for generally plotting graphs."""

from __future__ import annotations
from typing import Tuple

import geopandas as gpd
import networkx as nx
import shapely.geometry

from src.constants import PREFERRED_CRS


def create_node_edge_geometries(
    graph: nx.Graph, crs: str = PREFERRED_CRS
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create node and edge geometries for the networkx graph G.

    Returns node and edge geometries in two GeoDataFrames. The output can be used for
    plotting a graph.

    Args:
        graph (nx.Graph): graph with nodes and edges
        crs (str, optional): coordinate reference system. Defaults to UTM35N.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: dataframes of nodes and edges
            respectively.
    """

    node_gdf = gpd.GeoDataFrame(columns=["id", "geometry"])
    rep_points = graph.nodes(data="rep_point")
    for idx, rep_point in rep_points:
        node_gdf.loc[idx] = [idx, rep_point]

    edge_gdf = gpd.GeoDataFrame(columns=["id", "geometry"])
    for idx, (node_a, node_b) in enumerate(graph.edges()):
        point_a = rep_points[node_a]
        point_b = rep_points[node_b]
        line = shapely.geometry.LineString([point_a, point_b])

        edge_gdf.loc[idx] = [idx, line]

    node_gdf = node_gdf.set_crs(crs)
    edge_gdf = edge_gdf.set_crs(crs)

    return node_gdf, edge_gdf
