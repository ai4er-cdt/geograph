"""Module for processing and analysis of the geospatial graph."""

import bz2
import gzip
import os
import pathlib
import pickle
from itertools import zip_longest
from typing import Dict, List, Optional, Union

import affine
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree
from src.constants import PROJECT_PATH
from src.data_loading.rasterio_utils import polygonise
from tqdm import tqdm


class GeoGraph:
    """Class for the fragmentation graph."""

    def __init__(
        self,
        dataframe: Optional[gpd.GeoDataFrame] = None,
        attributes: Optional[List[str]] = None,
        graph_load_path: Optional[Union[str, os.PathLike]] = None,
        graph_save_path: Optional[Union[str, os.PathLike]] = None,
        raster_array: Optional[np.ndarray] = None,
        raster_load_path: Optional[Union[str, os.PathLike]] = None,
        raster_save_path: Optional[Union[str, os.PathLike]] = None,
        vector_path: Optional[Union[str, os.PathLike]] = None,
        tolerance: float = 0.0,
        mask: Optional[np.ndarray] = None,
        transform: affine.Affine = affine.identity,
        crs: Optional[str] = None,
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
            attributes (Optional[List[str]], optional): columns of dataframe
            that are added to nodes of graph as attributes. Defaults to None,
            which will cause all attributes to be added.
            graph_load_path (str or pathlib.Path, optional): A path to a pickle
            file to load from, can be `.gz` or `.bz2`. Defaults to None.
            graph_save_path (str or pathlib.Path, optional): A path to a pickle
            file to save the graph to, can be `.gz` or `.bz2`. Defaults to None,
            which will save the graph in the project root in the `data`
            folder.
            raster_array (np.ndarray, optional): 2D numpy array with the raster
            data that can be passed as an alternative to `raster_path`.
            vector_path (str or pathlib.Path, optional): A path to a file of
            vector data, either in GPKG or Shapefile format. Defaults to None.
            raster_load_path (str or pathlib.Path, optional): A path to a file
            of raster data in GeoTiff format. Defaults to None.
            raster_save_path (str or pathlib.Path, optional): A path to a file
            to save the polygonised raster data in. A path to a GPKG file is
            recommended, but Shapefiles also work. Defaults to None, which saves
            to the project root in the `data` folder in GPKG format.
            tolerance (float, optional): Adds edges between neighbours that are
            at most `tolerance` metres apart. Defaults to 0.
            mask (np.ndarray, optional): Boolean mask that can be applied over
            the polygonisation. Defaults to None.
            transform (affine.Affine, optional): Affine transformation to apply
            when polygonising. Defaults to the identity transform.
            crs (str, optional): Coordinate reference system to set on the
            resulting dataframe. Defaults to None.
        """
        super().__init__()
        self._graph = nx.Graph()
        if graph_save_path is None:
            graph_save_path = PROJECT_PATH / "data" / "geograph.bz2"
            os.makedirs(graph_save_path.parent, exist_ok=True)
        if raster_save_path is None:
            raster_save_path = PROJECT_PATH / "data" / "geo_data.gpkg"
            os.makedirs(raster_save_path.parent, exist_ok=True)
        self.tolerance: float = tolerance

        if graph_load_path is not None:
            self._rtree = self._load_graph(pathlib.Path(graph_load_path))
        elif dataframe is not None:
            self._rtree = self._dataframe_to_graph(
                dataframe, attributes, tolerance=self.tolerance
            )
            self._save_graph(pathlib.Path(graph_save_path))
        elif raster_array is not None or raster_load_path is not None:
            self._rtree = self._load_raster(
                raster_load_path,
                pathlib.Path(raster_save_path),
                array=raster_array,
                mask=mask,
                transform=transform,
                crs=crs,
            )
            self._save_graph(pathlib.Path(graph_save_path))
        elif vector_path is not None:
            self._rtree = self._load_vector(pathlib.Path(vector_path), attributes)
            self._save_graph(pathlib.Path(graph_save_path))

        print(
            f"Graph successfully loaded with {self.graph.number_of_nodes()} nodes",
            f"and {self.graph.number_of_edges()} edges",
        )

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

    def _load_vector(
        self,
        vector_path: pathlib.Path,
        attributes: Optional[List[str]] = None,
        load_slice=None,
    ) -> STRtree:
        """
        Load graph and rtree with vector data from GeoPackage or shape file.

        Args:
            vector_path (pathlib.Path): Path to a gpkg or shp file.
            attributes (Optional[List[str]], optional): columns of the dataframe
            that are added to nodes of graph as attributes. Defaults to None,
            which will cause all attributes to be added.
            load_slice: A slice object denoting the rows of the dataframe to
            load. Defaults to None, meaning load all rows.

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
        return self._dataframe_to_graph(
            dataframe, attributes=attributes, tolerance=self.tolerance
        )

    def _load_raster(
        self,
        raster_path: Optional[Union[str, os.PathLike]],
        save_path: pathlib.Path,
        array: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        transform: affine.Affine,
        crs: Optional[str],
    ) -> STRtree:
        """
        Load raster data, polygonise, then load graph and rtree.

        The raster data should be in GeoTiff format.

        Args:
            raster_path (pathlib.Path): A path to a file of raster data in
            GeoTiff format.
            save_path (pathlib.Path, optional): A path to a file to save the
            polygonised raster data in. A path to a GPKG file is recommended,
            but Shapefiles also work.
            array (np.ndarray, optional): 2D numpy array with the raster data
            that can be passed as an alternative to `raster_path`.
            mask (np.ndarray, optional): Boolean mask that can be applied over
            the polygonisation. Defaults to None.
            transform (affine.Affine, optional): Affine transformation to apply
            when polygonising. Defaults to the identity transform.
            crs (str, optional): Coordinate reference system to set on the
            resulting dataframe. Defaults to None.

        Raises:
            ValueError: If `save_path` is not a GPKG or Shapefile, or if both
            `raster_path` and `array` are None.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        if save_path.suffix not in (".shp", ".gpkg"):
            raise ValueError("Argument `save_path` should be a GPKG or Shapefile.")
        if array is None and raster_path is None:
            raise ValueError("One of `raster_path` or `array` arguments must be used.")
        elif array is None and raster_path is not None:
            with rasterio.Env():
                with rasterio.open(raster_path) as image:
                    data = image.read(1)  # Read band 1
        else:
            data = array
        vector_df = polygonise(
            data_array=data,
            mask=mask,
            connectivity=4,
            apply_buffer=True,
            transform=transform,
            crs=crs,
        )

        if save_path.suffix == ".gpkg":
            vector_df.to_file(save_path, driver="GPKG")
        else:
            vector_df.to_file(save_path)
        return self._dataframe_to_graph(
            vector_df, attributes=["geometry", "class_label"], tolerance=self.tolerance
        )

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
        self,
        df: gpd.GeoDataFrame,
        attributes: Optional[List[str]] = None,
        tolerance: float = 0.0,
    ) -> STRtree:
        """
        Convert geopandas dataframe to networkx graph.

        This code takes around 3 minutes to run on JASMIN for the Chernobyl
        habitat data.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing polygon objects from
            a shape file.
            attributes (Optional[List[str]], optional): columns of df that are
            added to nodes of graph as attributes. Defaults to None,
            which will cause all attributes to be added.
            tolerance (float, optional): Adds edges between neighbours that are
            at most `tolerance` metres apart. Defaults to 0.

        Raises:
            ValueError: If `tolerance` < 0.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        if tolerance < 0.0:
            raise ValueError("`tolerance` must be greater than 0.")
        # If no attribute list is given, all
        # columns in df are used
        if attributes is None:
            attributes = df.columns.tolist()
        elif not set(attributes).issubset(set(df.columns)):
            raise ValueError("`attributes` must only contain column names in `df`.")

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
            if tolerance > 0:
                # Expand the borders of the polygon by `tolerance```
                new_polygon = polygon.buffer(tolerance)
                # find the indexes of all polygons which intersect with this one
                neighbours: List[int] = [
                    id_dict[id(nbr)]
                    for nbr in tree.query(new_polygon)
                    if nbr.intersects(new_polygon)
                    and id_dict[id(nbr)] != id_dict[id(polygon)]
                ]
            else:
                neighbours = [
                    id_dict[id(nbr)]
                    for nbr in tree.query(polygon)
                    if nbr.intersects(polygon)
                    and id_dict[id(nbr)] != id_dict[id(polygon)]
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
                **row_attributes,
            )

        # iterate through the dict and add edges between neighbouring polygons
        for polygon_id, neighbours in tqdm(
            graph_dict.items(), desc="Step 2 of 2: Adding edges"
        ):
            for neighbour_id in neighbours:
                self.graph.add_edge(polygon_id, neighbour_id)

        return tree

    def merge_nodes(
        self, node_list: List[int], class_label: int, final_index: int = None
    ) -> None:
        """
        Merge a list of nodes in the graph together into a single node.

        This will create a node with a neighbour list and polygon which is the
        union of the nodes in `node_list`.

        Args:
            node_list (List[int]): List of integer node indexes in the graph.
            class_label (int): Class label for the resulting node.
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
            geometry=polygon,
            class_label=class_label,
        )
        self.graph.add_edges_from(
            zip_longest([final_index], adjacency_list, fillvalue=final_index)
        )
