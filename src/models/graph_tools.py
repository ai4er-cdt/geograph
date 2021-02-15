"""
This module contains the functions to build a networkx graph from polygon habitat data.
"""


from typing import Optional, List

import networkx as nx
import geopandas as gpd
from shapely.strtree import STRtree
from tqdm import tqdm


def create_nx_graph(
    gdf: gpd.GeoDataFrame, attributes: Optional[List[str]] = None
) -> nx.Graph:
    """Create networkx graph from polygons in GeoDataFrame.

    For each polygon in gdf a node (=vertex) is created. Edges are created when the
    polygons of two nodes overlap or touch borders. Largely taken from a notebook
    by @herbiebradley.

    Args:
        gdf (gpd.GeoDataFrame): data frame of polygons
        attributes (Optional[List[str]], optional): columns of gdf that are added to
            nodes of graph as attributes. Defaults to None.

    Returns:
        nx.Graph: graph as described above
    """

    # If no attribute list is given, all
    # columns in gdf are used
    if attributes is None:
        attributes = gdf.columns.tolist()

    geom = gdf.geometry.tolist()
    # Build dict mapping hashable unique object ids for each polygon
    # to their index in geom
    id_dict = {id(poly): index for index, poly in enumerate(geom)}
    # Build Rtree from geometry
    tree = STRtree(geom)

    # Initialising graph
    graph_dict = {}
    graph = nx.Graph()

    # Creating nodes (=vertices) and finding neighbors
    for index, polygon in tqdm(
        enumerate(geom),
        desc="Step 1 of 2: Creating nodes and finding neighbours",
        total=len(geom),
    ):
        # find the indexes of all polygons which touch the borders of or overlap
        # with this one
        neighbours = [
            id_dict[id(nbr)]
            for nbr in tree.query(polygon)
            if nbr.touches(polygon) or nbr.overlaps(polygon)
        ]
        # this dict maps polygon indices in gdf to a list
        # of neighbouring polygon indices
        graph_dict[index] = neighbours
        row = gdf.loc[index]
        # getting dict of column values in row
        row_attributes = dict(zip(attributes, [row[attr] for attr in attributes]))
        # add each polygon as a node to the graph with all attributes
        rep_point = row.geometry.representative_point()
        graph.add_node(index, representative_point=rep_point, **row_attributes)

    # iterate through the dict and add all edges between neighbouring polygons
    for polygon_id, neighbours in tqdm(
        graph_dict.items(), desc="Step 2 of 2: Adding edges"
    ):
        for neighbour_id in neighbours:
            graph.add_edge(polygon_id, neighbour_id)

    return graph
