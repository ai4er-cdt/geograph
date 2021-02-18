"""Module for processing and analysis of the graph generated from shp files."""

import os
import pathlib
from typing import Dict, List, Optional, Union

import geopandas as gpd
import networkx as nx
from shapely.strtree import STRtree
from tqdm import tqdm


class GeoGraph:
    """Class for the fragmentation graph."""

    def __init__(
        self,
        dataframe: Optional[gpd.GeoDataFrame] = None,
        graph_path: Optional[Union[str, os.PathLike]] = None,
        vector_path: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        """
        Class for the fragmentation graph.

        This class can load a pickled networkx graph directly, or create the
        graph from a shape file or dataframe.

        Args:
            dataframe (gpd.GeoDataFrame, optional): A geopandas dataframe
            object. Defaults to None.
            graph_path (str or pathlib.Path, optional): A path to a pickled
            networkx graph. Defaults to None.
            vector_path (str or pathlib.Path, optional): A path to a file of
            vector data, either in GPKG or shapefile format. Defaults to None.
        """
        super().__init__()
        self._graph = nx.Graph()
        if graph_path is not None:
            self._load_pickle(pathlib.Path(graph_path))
        elif vector_path is not None:
            self._load_vector(pathlib.Path(vector_path))
        else:
            self._dataframe_to_graph(dataframe)

    @property
    def graph(self) -> nx.Graph:
        """Return networkx graph object."""
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        self._graph = new_graph

    def _load_vector(self, vector_path: pathlib.Path) -> None:
        """
        Load GeoDataFrame with vector data from GeoPackage or shape file.

        Args:
            vector_path (pathlib.Path): Path to a gpkg or shp file.
        """
        if vector_path.suffix not in (".shp", ".gpkg"):
            raise ValueError("Argument `vector_path` should be a GPKG or shapefile.")
        # First try to load as GeoPackage, then as Shapefile.
        dataframe = gpd.read_file(
            vector_path, enabled_drivers=["GPKG", "ESRI Shapefile"]
        )
        self._dataframe_to_graph(dataframe)

    def _load_pickle(self, graph_path: pathlib.Path) -> None:
        """
        Load networkx graph object from a pickle file.

        Args:
            graph_path (pathlib.Path): Path to a pickle file.
        """
        if graph_path.suffix not in (".pickle", ".pkl"):
            raise ValueError("Argument `graph_path` should be a pickle file.")
        self.graph = nx.read_gpickle(graph_path)

    def _dataframe_to_graph(
        self, df: gpd.GeoDataFrame, attributes: Optional[List[str]] = None
    ) -> None:
        """
        Convert geopandas dataframe to networkx graph.

        This code takes around 3 minutes to run on JASMIN for the Chernobyl
        habitat data.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing polygon objects from
            a shape file.
            attributes (Optional[List[str]], optional): columns of gdf that are
            added to nodes of graph as attributes. Defaults to None.
        """
        # If no attribute list is given, all
        # columns in df are used
        if attributes is None:
            attributes = df.columns.tolist()

        geom = df.geometry.tolist()
        # Build dict mapping hashable unique object ids for each polygon
        # to their index in geom
        id_dict: Dict[int, int] = {
            id(polygon): index for index, polygon in enumerate(geom)
        }
        # Build Rtree from geometry
        tree = STRtree(geom)
        # this dict maps polygon indices in df to a list
        # of neighbouring polygon indices
        graph_dict = {}

        # Creating nodes (=vertices) and finding neighbors
        for index, polygon in tqdm(
            enumerate(geom),
            desc="Step 1 of 2: Creating nodes and finding neighbours",
            total=len(geom),
        ):
            # find the indexes of all polygons which touch the borders of or
            # overlap with this one
            neighbours: List[int] = [
                id_dict[id(nbr)]
                for nbr in tree.query(polygon)
                if nbr.intersects(polygon) and id_dict[id(nbr)] != id_dict[id(polygon)]
            ]
            # this dict maps polygon indices in df to a list
            # of neighbouring polygon indices
            graph_dict[index] = neighbours
            row = df.loc[index]
            # getting dict of column values in row
            row_attributes = dict(zip(attributes, [row[attr] for attr in attributes]))
            # add each polygon as a node to the graph with all attributes
            self.graph.add_node(
                index,
                rep_point=polygon.representative_point(),
                area=polygon.area,
                perimeter=polygon.length,
                **row_attributes
            )

        # iterate through the dict and add edges between neighbouring polygons
        for polygon_id, neighbours in tqdm(
            graph_dict.items(), desc="Step 2 of 2: Adding edges"
        ):
            for neighbour_id in neighbours:
                self.graph.add_edge(polygon_id, neighbour_id)
