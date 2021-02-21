"""Module for processing and analysis of the graph generated from shp files."""

import bz2
import gzip
import os
import pathlib
import pickle
from itertools import zip_longest
from typing import Dict, List, Optional, Union

import geopandas as gpd
import networkx as nx
import rasterio
import shapely
from rasterio.features import shapes
from shapely.ops import unary_union
from shapely.strtree import STRtree
from src.constants import PROJECT_PATH
from tqdm import tqdm


class GeoGraph:
    """Class for the fragmentation graph."""

    def __init__(
        self,
        dataframe: Optional[gpd.GeoDataFrame] = None,
        graph_load_path: Optional[Union[str, os.PathLike]] = None,
        graph_save_path: Optional[Union[str, os.PathLike]] = None,
        raster_load_path: Optional[Union[str, os.PathLike]] = None,
        raster_save_path: Optional[Union[str, os.PathLike]] = None,
        vector_path: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        """
        Class for the fragmentation graph.

        This class can load a pickled networkx graph directly, or create the
        graph from vector data, raster data, or a dataframe containing polygons.

        Args:
            dataframe (gpd.GeoDataFrame, optional): A geopandas dataframe
            object containing a `geometry` column with polygon data. To load
            polygon class labels as node attributes, include these in a column.
            Defaults to None.
            graph_load_path (str or pathlib.Path, optional): A path to a pickle
            file to load from, can be `.gz` or `.bz2`. Defaults to None.
            graph_save_path (str or pathlib.Path, optional): A path to a pickle
            file to save the graph to, can be `.gz` or `.bz2`. Defaults to None,
            which will save the graph in the project root in the `data`
            folder.
            vector_path (str or pathlib.Path, optional): A path to a file of
            vector data, either in GPKG or Shapefile format. Defaults to None.
            raster_load_path (str or pathlib.Path, optional): A path to a file
            of raster data in GeoTiff format. Defaults to None.
            raster_save_path (str or pathlib.Path, optional): A path to a file
            to save the polygonised raster data in. A path to a GPKG file is
            recommended, but Shapefiles also work. Defaults to None, which saves
            to the project root in the `data` folder in GPKG format.
        """
        super().__init__()
        self._graph = nx.Graph()
        if graph_save_path is None:
            graph_save_path = PROJECT_PATH / "data" / "geograph.bz2"
            os.makedirs(graph_save_path.parent, exist_ok=True)
        if raster_save_path is None:
            raster_save_path = PROJECT_PATH / "data" / "geo_data.gpkg"
            os.makedirs(raster_save_path.parent, exist_ok=True)

        if graph_load_path is not None:
            self._rtree = self._load_graph(pathlib.Path(graph_load_path))
        elif dataframe is not None:
            self._rtree = self._dataframe_to_graph(dataframe)
            self._save_graph(pathlib.Path(graph_save_path))
        elif vector_path is not None:
            self._rtree = self._load_vector(pathlib.Path(vector_path))
            self._save_graph(pathlib.Path(graph_save_path))
        elif raster_load_path is not None:
            self._rtree = self._load_raster(
                pathlib.Path(raster_load_path), pathlib.Path(raster_save_path)
            )
            self._save_graph(pathlib.Path(graph_save_path))

    @property
    def graph(self) -> nx.Graph:
        """Return networkx graph object."""
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        self._graph = new_graph

    @property
    def rtree(self) -> STRtree:
        """Return Rtree object."""
        return self._rtree

    def _load_vector(self, vector_path: pathlib.Path, load_slice=None) -> STRtree:
        """
        Load graph and rtree with vector data from GeoPackage or shape file.

        Args:
            vector_path (pathlib.Path): Path to a gpkg or shp file.
            load_slice : A slice object denoting the rows of the dataframe to load.
            Defaults to None, meaning load all rows.

        Raises:
            ValueError: If `vector_path` is not a GPKG or Shapefile.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        if vector_path.suffix not in (".shp", ".gpkg"):
            raise ValueError("Argument `vector_path` should be a GPKG or Shapefile.")
        # First try to load as GeoPackage, then as Shapefile.
        if slice is not None:
            dataframe = gpd.read_file(
                vector_path, rows=load_slice, enabled_drivers=["GPKG", "ESRI Shapefile"]
            )
        else:
            dataframe = gpd.read_file(
                vector_path, enabled_drivers=["GPKG", "ESRI Shapefile"]
            )
        return self._dataframe_to_graph(dataframe)

    def _load_raster(
        self, raster_path: pathlib.Path, save_path: pathlib.Path
    ) -> STRtree:
        """
        Load raster data, polygonize, then load graph and rtree.

        The raster data should be in GeoTiff format.

        Args:
            raster_path (pathlib.Path): A path to a file of raster data in
            GeoTiff format.
            save_path (pathlib.Path, optional): A path to a file to save the
            polygonised raster data in. A path to a GPKG file is recommended,
            but Shapefiles also work.

        Raises:
            ValueError: If `save_path` is not a GPKG or Shapefile.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        if save_path.suffix not in (".shp", ".gpkg"):
            raise ValueError("Argument `save_path` should be a GPKG or Shapefile.")
        mask = None
        with rasterio.Env():
            with rasterio.open(raster_path) as image:
                image_band_1 = image.read(1)
                geoms = [
                    {"properties": {"raster_val": v}, "geometry": s}
                    for _, (s, v) in enumerate(
                        shapes(
                            image_band_1,
                            mask=mask,
                            connectivity=4,
                            transform=image.transform,
                        )
                    )
                ]
        vector_df = gpd.GeoDataFrame.from_features(geoms)
        # Redraw geometries to ensure polygons are valid.
        vector_df.geometry = vector_df.geometry.buffer(0)

        if save_path.suffix == ".gpkg":
            vector_df.to_file(save_path, driver="GPKG")
        else:
            vector_df.to_file(save_path)
        return self._dataframe_to_graph(vector_df, attributes=["geometry"])

    def _load_graph(self, graph_path: pathlib.Path) -> STRtree:
        """
        Load networkx graph and rtree objects from a pickle file.

        Args:
            graph_path (pathlib.Path): Path to a pickle file. Can be compressed
            with gzip or bz2.

        Raises:
            ValueError: If `graph_path` is not a pickle, bz2, or gz file.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        if graph_path.suffix not in (".pickle", ".pkl", ".gz", ".bz2"):
            raise ValueError("Argument `graph_path` should be a pickle file.")

        if graph_path.suffix == ".bz2":
            bz2_file = bz2.BZ2File(graph_path, "rb")
            data = pickle.load(bz2_file)
            bz2_file.close()
        elif graph_path.suffix == ".gz":
            gz_file = gzip.GzipFile(graph_path, "rb")
            gz_data = gz_file.read()
            data = pickle.loads(gz_data)
            gz_file.close()
        else:
            with open(graph_path, "rb") as file:
                data = pickle.load(file)

        self.graph = data["graph"]
        return data["rtree"]

    def _save_graph(self, save_path: pathlib.Path) -> None:
        """
        Save graph with attributes and rtree as pickle file. Can be compressed.

        Args:
            save_path (pathlib.Path): Path to a pickle file. Can be compressed
            with gzip or bz2 by passing filenames ending in `gz` or `bz2`.

        Raises:
            ValueError: If `save_path` is not a pickle, bz2, or gz file path.
        """
        if save_path.suffix not in (".pickle", ".pkl", ".gz", ".bz2"):
            raise ValueError(
                "Argument `save_path` should be a pickle file or compressed file."
            )

        data = {"graph": self.graph, "rtree": self.rtree}
        if save_path.suffix == ".bz2":
            with bz2.BZ2File(save_path, "wb") as bz2_file:
                pickle.dump(data, bz2_file)
        elif save_path.suffix == ".gz":
            gz_file = gzip.GzipFile(save_path, "wb")
            gz_file.write(pickle.dumps(data))
            gz_file.close()
        else:
            with open(save_path, "wb") as file:
                pickle.dump(data, file)

    def _dataframe_to_graph(
        self, df: gpd.GeoDataFrame, attributes: Optional[List[str]] = None
    ) -> STRtree:
        """
        Convert geopandas dataframe to networkx graph.

        This code takes around 3 minutes to run on JASMIN for the Chernobyl
        habitat data.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing polygon objects from
            a shape file.
            attributes (Optional[List[str]], optional): columns of gdf that are
            added to nodes of graph as attributes. Defaults to None.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        # If no attribute list is given, all
        # columns in df are used
        if attributes is None:
            attributes = df.columns.tolist()

        geom: Dict[int, shapely.Polygon] = df.geometry.to_dict()
        # Build dict mapping hashable unique object ids for each polygon
        # to their index in geom
        id_dict: Dict[int, int] = {
            id(polygon): index for index, polygon in geom.items()
        }
        # Build Rtree from geometry
        tree = STRtree(geom.values())
        # this dict maps polygon indices in df to a list
        # of neighbouring polygon indices
        graph_dict = {}

        # Creating nodes (=vertices) and finding neighbors
        for index, polygon in tqdm(
            geom.items(),
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

        return tree

    def merge_nodes(self, node_list: List[int], final_index: int = None) -> None:
        """
        Merge a list of nodes in the graph together into a single node.

        This will create a node with a neighbour list and polygon which is the
        union of the nodes in `node_list`.

        Args:
            node_list (List[int]): List of integer node indexes in the graph.
            final_index (int, optional): Index to assign to the resulting node.
            Defaults to None, in which case it becomes the number of nodes in
            the graph.

        Raises:
            ValueError: If there are invalid nodes in `node_list`, or if
            `final_index` is an existing node not in `node_list`.
        """
        if not all(self.graph.has_node(node) for node in node_list):
            raise ValueError("`node_list` must only contain valid nodes.")
        if (
            final_index is not None
            and final_index not in node_list
            and self.graph.has_node(final_index)
        ):
            raise ValueError(
                "`final_index` must not be an existing node not in `node_list`."
            )
        if final_index is None:
            final_index = self.graph.number_of_nodes
        adjacency_list = set()
        for node in node_list:
            adjacency_list.update(self.graph.neighbors(node))
        polygon = unary_union(
            [self.graph.nodes[node]["geometry"] for node in node_list]
        )
        self.graph.remove_nodes_from(node_list)
        self.graph.add_node(
            final_index,
            rep_point=polygon.representative_point(),
            area=polygon.area,
            perimeter=polygon.length,
        )
        self.graph.add_edges_from(
            zip_longest([final_index], adjacency_list, fillvalue=final_index)
        )
