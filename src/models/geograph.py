"""
Module for processing and analysis of the geospatial graph.

See https://networkx.org/documentation/stable/index.html for graph operations.
"""

import bz2
import gzip
import os
import pathlib
import pickle
from itertools import zip_longest
from typing import Dict, List, Optional, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree
from src.data_loading.rasterio_utils import polygonise
from tqdm import tqdm

VALID_EXTENSIONS = (
    ".pickle",
    ".pkl",
    ".gz",
    ".bz2",
    ".shp",
    ".gpkg",
    ".tiff",
    ".tif",
    ".geotif",
    ".geotiff",
)


class GeoGraph:
    """Class for the fragmentation graph."""

    def __init__(
        self,
        data,
        attributes: Optional[List[str]] = None,
        graph_save_path: Optional[Union[str, os.PathLike]] = None,
        raster_save_path: Optional[Union[str, os.PathLike]] = None,
        tolerance: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Class for the fragmentation graph.

        This class can load a pickled networkx graph directly, or create the
        graph from a path to vector data, a path to raster data, a numpy array
        containing raster data, or a dataframe containing polygons.

        Note that when loading vector data, the class label column may not be
        named "class_label", which will cause an error. If loading from a
        dataframe, it must contain a column with this name. Dataframes must also
        contain a "geometry" column containing the polygon data.

        Warning: loading and saving GeoGraphs uses pickle. Loading untrusted
        data using the pickle module is not secure as it can execute arbitrary
        code. Therefore, only load GeoGraphs that come from a trusted source.
        See the pickle documentation for more details:
        https://docs.python.org/3/library/pickle.html

        Args:
            data: Can be a path to a pickle file or compressed pickle file to load
            the graph from, a path to vector data in GPKG or Shapefile format,
            a path to raster data in GeoTiff format, a numpy array containing raster
            data, or a dataframe containing polygons.
            attributes (Optional[List[str]], optional): Columns of the dataframe
            that are added to nodes of graph as attributes. Defaults to None,
            which will cause all attributes to be added.
            graph_save_path (str or pathlib.Path, optional): A path to a pickle
            file to save the graph to, can be `.gz` or `.bz2`. Defaults to None,
            which will not save the graph.
            raster_save_path (str or pathlib.Path, optional): A path to a file
            to save the polygonised raster data in. A path to a GPKG file is
            recommended, but Shapefiles also work. Defaults to None, which will
            not save the polygonised data.
            tolerance (float, optional): Adds edges between neighbours that are
            at most `tolerance` units apart. Defaults to 0.

            **mask (np.ndarray, optional): Boolean mask that can be applied over
            the polygonisation. Defaults to None.
            **transform (affine.Affine, optional): Affine transformation to
            apply when polygonising. Defaults to the identity transform.
            **crs (str, optional): Coordinate reference system to set on the
            resulting dataframe. Defaults to None.
            **connectivity (int, optional): Use 4 or 8 pixel connectivity for
            grouping pixels into features. 8 can cause issues, Defaults to 4.
            **apply_buffer (bool, optional): Apply shapely buffer function to
            the polygons after polygonising. This can fix issues with the
            polygonisation creating invalid geometries.
        """
        super().__init__()
        self._graph = nx.Graph()

        if raster_save_path is not None:
            raster_save_path = pathlib.Path(raster_save_path)
            if raster_save_path.suffix not in (".shp", ".gpkg"):
                raise ValueError("Argument `save_path` should be a GPKG or Shapefile.")
            os.makedirs(raster_save_path.parent, exist_ok=True)
        self.tolerance: float = tolerance
        load_from_graph: bool = False

        # Load from disk
        if isinstance(data, (str, os.PathLike)):
            load_path = pathlib.Path(data)
            assert load_path.exists()
            # Load from saved graph
            if load_path.suffix in (".pickle", ".pkl", ".gz", ".bz2"):
                self._rtree = self._load_from_graph_path(load_path)
                load_from_graph = True
            # Load from saved vector data
            elif load_path.suffix in (".shp", ".gpkg"):
                self._rtree = self._load_from_vector_path(load_path, attributes)
            # Load from saved raster data
            elif load_path.suffix in (".tiff", ".tif", ".geotif", ".geotiff"):
                self._rtree = self._load_from_raster_path(
                    load_path, raster_save_path, **kwargs
                )
            else:
                raise ValueError(
                    f"""Extension {load_path.suffix} unknown.
                                 Must be one of {VALID_EXTENSIONS}"""
                )

        # Load from objects in memory
        # Load from dataframe
        elif isinstance(data, gpd.GeoDataFrame):
            self._rtree = self._load_from_dataframe(
                data, attributes, tolerance=self.tolerance
            )
        # Load from raster array
        elif isinstance(data, np.ndarray):
            self._rtree = self._load_from_raster(data, raster_save_path, **kwargs)
        # Save resulting graph, if we didn't load from graph.
        if not load_from_graph and graph_save_path is not None:
            graph_save_path = pathlib.Path(graph_save_path)
            if graph_save_path.suffix not in (".pickle", ".pkl", ".gz", ".bz2"):
                raise ValueError(
                    """Argument `graph_save_path` should be a pickle file or
                    compressed file."""
                )
            os.makedirs(graph_save_path.parent, exist_ok=True)
            self._save_graph(graph_save_path)

        print(
            f"Graph successfully loaded with {self.graph.number_of_nodes()} nodes",
            f"and {self.graph.number_of_edges()} edges.",
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

    def _load_from_vector_path(
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

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        # First try to load as GeoPackage, then as Shapefile.
        if slice is not None:
            dataframe = gpd.read_file(
                vector_path, rows=load_slice, enabled_drivers=["GPKG", "ESRI Shapefile"]
            )
        else:
            dataframe = gpd.read_file(
                vector_path, enabled_drivers=["GPKG", "ESRI Shapefile"]
            )
        return self._load_from_dataframe(
            dataframe, attributes=attributes, tolerance=self.tolerance
        )

    def _load_from_raster_path(
        self,
        raster_path: pathlib.Path,
        save_path: Optional[pathlib.Path],
        **raster_kwargs,
    ) -> STRtree:
        """
        Load raster data from a GeoTiff file, then load graph and rtree.

        Args:
            raster_path (pathlib.Path): A path to a file of raster data in
            GeoTiff format.
            save_path (pathlib.Path, optional): A path to a file to save the
            polygonised raster data in. A path to a GPKG file is recommended,
            but Shapefiles also work.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        with rasterio.Env():
            with rasterio.open(raster_path) as image:
                # Load band 1
                data = image.read(1)
        return self._load_from_raster(data, save_path, **raster_kwargs)

    def _load_from_raster(
        self, data_array: np.ndarray, save_path: Optional[pathlib.Path], **raster_kwargs
    ) -> STRtree:
        """
        Load raster data, polygonise, then load graph and rtree.

        The raster data should be in GeoTiff format.
        Polygonisation via `rasterio.features.shapes`, which uses `gdal_polygonize`.

        References:
        (1) https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html
        (2) https://gdal.org/programs/gdal_polygonize.html

        Args:
            data_array (np.ndarray): 2D numpy array with the raster data.
            save_path (pathlib.Path, optional): A path to a file to save the
            polygonised raster data in. A path to a GPKG file is recommended,
            but Shapefiles also work.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        vector_df = polygonise(data_array=data_array, **raster_kwargs)
        if save_path is not None:
            if save_path.suffix == ".gpkg":
                vector_df.to_file(save_path, driver="GPKG")
            else:
                vector_df.to_file(save_path, driver="ESRI Shapefile")
        return self._load_from_dataframe(
            vector_df, attributes=["geometry", "class_label"], tolerance=self.tolerance
        )

    def _load_from_graph_path(self, graph_path: pathlib.Path) -> STRtree:
        """
        Load networkx graph and rtree objects from a pickle file.

        Args:
            graph_path (pathlib.Path): Path to a pickle file. Can be compressed
            with gzip or bz2.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        if graph_path.suffix == ".bz2":
            with bz2.BZ2File(graph_path, "rb") as bz2_file:
                data = pickle.load(bz2_file)
        elif graph_path.suffix == ".gz":
            with gzip.GzipFile(graph_path, "rb") as gz_file:
                data = pickle.loads(gz_file.read())
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
        """
        data = {"graph": self.graph, "rtree": self.rtree}
        if save_path.suffix == ".bz2":
            with bz2.BZ2File(save_path, "wb") as bz2_file:
                pickle.dump(data, bz2_file)
        elif save_path.suffix == ".gz":
            with gzip.GzipFile(save_path, "wb") as gz_file:
                gz_file.write(pickle.dumps(data))
        else:
            with open(save_path, "wb") as file:
                pickle.dump(data, file)

    def _load_from_dataframe(
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
            at most `tolerance` units apart. Defaults to 0.

        Raises:
            ValueError: If `tolerance` < 0, if `class_label` is not a column in
            the dataframe, or if `attributes` contains column names not in the
            dataframe.

        Returns:
            STRtree: The Rtree object used to build the graph.
        """
        if tolerance < 0.0:
            raise ValueError("`tolerance` must be greater than 0.")
        if (
            attributes is not None
            and "class_label" not in attributes
            or "class_label" not in df.columns
        ):
            raise ValueError("`class_label` must be a column in the dataframe.")
        if (
            attributes is not None
            and "geometry" not in attributes
            or "geometry" not in df.columns
        ):
            raise ValueError("`geometry` must be a column in the dataframe.")
        # If no attribute list is given, all
        # columns in df are used
        if attributes is None:
            attributes = df.columns.tolist()
        elif not set(attributes).issubset(df.columns):
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
                    if id_dict[id(nbr)] != id_dict[id(polygon)]
                    and nbr.intersects(new_polygon)
                ]
            else:
                neighbours = [
                    id_dict[id(nbr)]
                    for nbr in tree.query(polygon)
                    if id_dict[id(nbr)] != id_dict[id(polygon)]
                    and nbr.intersects(polygon)
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
