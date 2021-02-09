"""
This file contains the functions to build an EcoGraph.
"""

from shapely.strtree import STRtree
import networkx as nx


def create_nx_graph(gpd_df, attributes=None):
    """ Creates networkx (nx) graph from GeoDataFrame.
    Largely taken from a notebook by @herbiebradley.
    """
    
    # If no attribute list is given, all
    # columns in gpd_df are used
    if not attributes:
        attributes = gpd_df.columns.tolist()
    
    geom = gpd_df.geometry.tolist()
    # Build dict mapping hashable unique object ids for each polygon to their index in geom
    id_dict = {id(poly):index for index, poly in enumerate(geom)}
    # Build Rtree from geometry
    tree = STRtree(geom)

    # Initialising graph
    graph_dict = {}
    G = nx.Graph()

    # Creating nodes (=vertices) and finding neighbors
    for index, polygon in enumerate(geom):
        # find the indexes of all polygons which touch the borders of or overlap with this one
        neighbours = [id_dict[id(nbr)] for nbr in tree.query(polygon) if nbr.touches(polygon) or nbr.overlaps(polygon)]
        # this dict maps polygon indices in gpd_df to a list of neighbouring polygon indices
        graph_dict[index] = neighbours
        row = gpd_df.loc[index]
        # getting dict of column values in row
        row_attributes = dict(zip(attributes,[row[attr] for attr in attributes]))
        # add each polygon as a node to the graph with all attributes
        G.add_node(index, **row_attributes)

    # iterate through the dict and add all edges between neighbouring polygons
    for polygon_id, neighbours in graph_dict.items():
        for neighbour_id in neighbours:
            G.add_edge(polygon_id, neighbour_id)
            
    return G