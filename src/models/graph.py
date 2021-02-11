"""Module for processing and analysis of the graph generated from shp files."""

from typing import Optional

import geopandas as gpd
import networkx as nx
from shapely.strtree import STRtree


class FragGraph:
    def __init__(self, dataframe: Optional[gpd.GeoDataFrame] = None, graph_path: Optional[str] = None) -> None:
        """
        Class for the fragmentation graph.

        The `graph_path` argument allows for loading a saved networkx graph.

        Args:
            dataframe (Optional[gpd.GeoDataFrame], optional): A geopandas
            dataframe object. Defaults to None.
            graph_path (Optional[str], optional): A path to a pickled networkx
            graph. Defaults to None.
        """
        super().__init__()
        self._G = nx.Graph()
        if graph_path is not None:
            self._load_graph(graph_path)
        else:
            self._dataframe_to_graph(dataframe)

    @property
    def G(self) -> nx.Graph:
        """Return networkx graph object."""
        return self._G

    @G.setter
    def G(self, new_graph):
        self._G = new_graph

    def _load_graph(self, graph_path: str) -> None:
        """
        Load networkx graph object from a pickle file.

        Args:
            graph_path (str): Path to a pickle file.
        """
        if not graph_path.endswith(('pickle', 'pkl')):
            raise ValueError("Argument `graph_path` should be a pickle file.")
        self.G = nx.read_gpickle(graph_path)

    def _dataframe_to_graph(self, df: gpd.GeoDataFrame) -> None:
        """
        Convert geopandas dataframe to networkx graph.

        This code takes around 3 minutes to run on JASMIN for the Chernobyl
        habitat data.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing polygon objects from
            a shape file.
        """
        geom = df.geometry.tolist()
        # build dict mapping hashable unique object ids for each polygon to their index in geom
        id_dict = {id(pgon): index for index, pgon in enumerate(geom)}
        # build Rtree from geometry
        tree = STRtree(geom)

        graph_dict = {}
        for index, pgon in enumerate(geom):
            # find the indexes of all polygons which touch the borders of or overlap with this one
            neighbours = [id_dict[id(nbr)] for nbr in tree.query(pgon) if nbr.touches(pgon) or nbr.overlaps(pgon)]
            # this dict maps polygon indices in veg_data to a list of neighbouring polygon indices
            graph_dict[index] = neighbours
            row = df.loc[index]
            # add each polygon as a node to the graph, with the polygon object as node attributes
            # TODO: add custom node attributes
            self.G.add_node(index, geometry=pgon)

        # iterate through the dict and add all edges between neighbouring polygons
        for polygon_id, neighbours in graph_dict.items():
            for neighbour_id in neighbours:
                self.G.add_edge(polygon_id, neighbour_id)
