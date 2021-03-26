"""
Module for processing and analysis of the geospatial graph.

See https://networkx.org/documentation/stable/index.html for graph operations.
"""
from __future__ import annotations

import bz2
import gzip
import inspect
import os
import pathlib
import pickle
from copy import deepcopy
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from shapely.prepared import prep
from tqdm import tqdm

from geograph import binary_graph_operations, metrics
from geograph.metrics import CLASS_METRICS_DICT, Metric
from geograph.utils import rasterio_utils

pd.options.mode.chained_assignment = None  # default='warn'

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
        crs: Optional[Union[str, pyproj.CRS]] = None,
        graph_save_path: Optional[Union[str, os.PathLike]] = None,
        raster_save_path: Optional[Union[str, os.PathLike]] = None,
        columns_to_rename: Optional[Dict[str, str]] = None,
        tolerance: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Class for the fragmentation graph.

        This class can load a pickled networkx graph directly, or create the
        graph from
            - a path to vector data (.shp, .gpkg)
            - a path to raster data  (.tif, .tiff, .geotif, .geotiff)
            - a numpy array containing raster data
            - a dataframe containing polygons.

        Note that the final dataframe must contain a class label column named
        "class_label" and a "geometry column containing the polygon data - the
        `columns_to_rename` argument allows for renaming columns to ensure this.

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
            crs (str): Coordinate reference system to set on the resulting
                dataframe. Warning: whatever units of distance the CRS uses will be
                the units of distance for all polygon calculations, including for
                the `tolerance` argument. Using a lat-long CRS can therefore result
                in incoherent output.
            graph_save_path (str or pathlib.Path, optional): A path to a pickle
                file to save the graph to, can be `.gz` or `.bz2`. Defaults to None,
                which will not save the graph.
            raster_save_path (str or pathlib.Path, optional): A path to a file
                to save the polygonised raster data in. A path to a GPKG file is
                recommended, but Shapefiles also work. Defaults to None, which will
                not save the polygonised data.
            columns_to_rename (Dict[str, str], optional): A dictionary mapping
                column names in the loaded dataframe with the new names of these
                columns. Use this to ensure that the dataframe has "class_label" and
                "geometry" columns. Defaults to None.
            tolerance (float, optional): Adds edges between neighbours that are
                at most `tolerance` units apart. Defaults to 0.
            **mask (np.ndarray, optional): Boolean mask that can be applied over
                the polygonisation. Defaults to None.
            **transform (affine.Affine, optional): Affine transformation to
                apply when polygonising. Defaults to the identity transform.
            **connectivity (int, optional): Use 4 or 8 pixel connectivity for
                grouping pixels into features. Defaults to 4.
            **apply_buffer (bool, optional): Apply shapely buffer function to
                the polygons after polygonising. This can fix issues with the
                polygonisation creating invalid geometries.
        """
        super().__init__()
        self.graph = nx.Graph()
        self.habitats: Dict[str, HabitatGeoGraph] = {}
        self._crs: Optional[Union[str, pyproj.CRS]] = crs
        self._columns_to_rename: Optional[Dict[str, str]] = columns_to_rename
        self._tolerance: float = tolerance
        self.metrics: Dict[str, Metric] = {}
        self.class_metrics: Dict[str, Dict[Union[str, int], Metric]] = {}

        if raster_save_path is not None:
            raster_save_path = pathlib.Path(raster_save_path)
            if raster_save_path.suffix not in (".shp", ".gpkg"):
                raise ValueError("Argument `save_path` should be a GPKG or Shapefile.")
            os.makedirs(raster_save_path.parent, exist_ok=True)
        load_from_graph: bool = False

        # Load from disk
        if isinstance(data, (str, os.PathLike)):
            load_path = pathlib.Path(data)
            assert load_path.exists()
            # Load from saved graph
            if load_path.suffix in (".pickle", ".pkl", ".gz", ".bz2"):
                self.df = self._load_from_graph_path(load_path)
                load_from_graph = True
            # Load from saved vector data
            elif load_path.suffix in (".shp", ".gpkg"):
                self.df = self._load_from_vector_path(load_path)
            # Load from saved raster data
            elif load_path.suffix in (".tiff", ".tif", ".geotif", ".geotiff"):
                self.df = self._load_from_raster_path(
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
            self.df = self._load_from_dataframe(data, tolerance=self._tolerance)
        # Load from raster array
        elif isinstance(data, np.ndarray):
            self.df = self._load_from_raster(data, raster_save_path, **kwargs)
        else:
            raise ValueError(
                """Type of `data` unknown. Must be a dataframe, numpy
                             array, or file path."""
            )

        # Save resulting graph, if we didn't load from graph.
        if not load_from_graph and graph_save_path is not None:
            graph_save_path = pathlib.Path(graph_save_path)
            os.makedirs(graph_save_path.parent, exist_ok=True)
            self.save_graph(graph_save_path)

        self.components = self.get_graph_components(calc_polygons=False)

        print(
            f"Graph successfully loaded with {self.graph.number_of_nodes()} nodes",
            f"and {self.graph.number_of_edges()} edges.",
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, GeoGraph):
            return False
        return nx.fast_could_be_isomorphic(self.graph, o.graph)

    @property
    def rtree(self):
        """Return Rtree object."""
        return self.df.sindex

    @property
    def crs(self):
        """Return crs of dataframe."""
        return self.df.crs

    @property
    def bounds(self):
        """Return bounds of entire graph"""
        return self.df.sindex.bounds

    @property
    def class_label(self):
        """Return class label of nodes directly from underlying numpy array.

        Note: Uses `iloc` type indexing.
        """
        return self.df["class_label"].values

    @property
    def classes(self) -> np.ndarray:
        """Return a list of the sorted, unique class labels in the graph"""
        return np.unique(self.df["class_label"].values)

    @property
    def geometry(self):
        """Return geometry of nodes from underlying numpy array.

        Note: Uses `iloc` type indexing.
        """
        return self.df["geometry"].values

    def _load_from_vector_path(
        self,
        vector_path: pathlib.Path,
        load_slice=None,
    ) -> gpd.GeoDataFrame:
        """
        Load graph and dataframe with vector data from GeoPackage or shape file.

        Args:
            vector_path (pathlib.Path): Path to a gpkg or shp file.
            load_slice: A slice object denoting the rows of the dataframe to
                load. Defaults to None, meaning load all rows.

        Returns:
            gpd.GeoDataFrame: The dataframe containing polygon objects.
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
        return self._load_from_dataframe(dataframe, tolerance=self._tolerance)

    def _load_from_raster_path(
        self,
        raster_path: pathlib.Path,
        save_path: Optional[pathlib.Path],
        **raster_kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Load raster data from a GeoTiff file, then load graph and dataframe.

        Note: Assumes that relevant data is stored in the first band (band 1)
        by default.

        Args:
            raster_path (pathlib.Path): A path to a file of raster data in
                GeoTiff format.
            save_path (pathlib.Path, optional): A path to a file to save the
                polygonised raster data in. A path to a GPKG file is recommended,
                but Shapefiles also work.

        Returns:
            gpd.GeoDataFrame: The dataframe containing polygon objects.
        """
        with rasterio.open(raster_path) as image:
            # Load band 1 (Assumes that landcover map is in first band by default)
            data = image.read(1)
        return self._load_from_raster(data, save_path, **raster_kwargs)

    def _load_from_raster(
        self, data_array: np.ndarray, save_path: Optional[pathlib.Path], **raster_kwargs
    ) -> gpd.GeoDataFrame:
        """
        Load raster data, polygonise, then load graph and dataframe.

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
            gpd.GeoDataFrame: The dataframe containing polygon objects.
        """
        vector_df = rasterio_utils.polygonise(data_array=data_array, **raster_kwargs)
        if save_path is not None:
            if save_path.suffix == ".gpkg":
                vector_df.to_file(save_path, driver="GPKG")
            else:
                vector_df.to_file(save_path, driver="ESRI Shapefile")
            save_path.chmod(0o664)
        return self._load_from_dataframe(vector_df, tolerance=self._tolerance)

    def _load_from_graph_path(self, graph_path: pathlib.Path) -> gpd.GeoDataFrame:
        """
        Load networkx graph and dataframe objects from a pickle file.

        Args:
            graph_path (pathlib.Path): Path to a pickle file. Can be compressed
                with gzip or bz2.

        Returns:
            gpd.GeoDataFrame: The dataframe containing polygon objects.
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
        return data["dataframe"]

    def save_graph(
        self,
        save_path: Union[str, pathlib.Path],
        overwrite: bool = False,
        pickle_protocol: int = pickle.DEFAULT_PROTOCOL,
    ) -> None:
        """
        Save graph with attributes and dataframe as pickle file. Can be compressed.

        Args:
            save_path (Union[pathlib.Path, str]): Path to a pickle file. Can be
                compressed with gzip or bz2 by passing filenames ending in `gz` or
                `bz2`.
            overwrite (bool, optional): If True, an existing file at `save_path`
                will be overwritten. Else throws an error. Defaults to False.
            pickle_protocol (int, optional): Selects the pickle protocol that is used
                for python object serealisation. Supported protocols are explained here:
                https://docs.python.org/3/library/pickle.html#data-stream-format
                Defaults to pickle.DEFAULT_PROTOCOL (4 in python 3.8).

        Raises:
            ValueError: If `save_path` is not a pickle, gz, or bz2 file.
        """
        save_path = pathlib.Path(save_path)
        if not overwrite and save_path.exists():
            raise UserWarning(
                f"A file already exists at {save_path}. To overwrite, ",
                "set the `overwrite` flag to True.",
            )

        if save_path.suffix not in (".pickle", ".pkl", ".gz", ".bz2"):
            raise ValueError(
                "Argument `save_path` should end in `.pickle`, `.pkl`, `.gz` or `.bz2` "
                "to indicate a pickle file or compressed pickle file."
            )
        data = {"graph": self.graph, "dataframe": self.df}
        if save_path.suffix == ".bz2":
            with bz2.BZ2File(save_path, "wb") as bz2_file:
                pickle.dump(data, bz2_file, protocol=pickle_protocol)
        elif save_path.suffix == ".gz":
            with gzip.GzipFile(save_path, "wb") as gz_file:
                gz_file.write(pickle.dumps(data, protocol=pickle_protocol))
        else:
            with open(save_path, "wb") as file:
                pickle.dump(data, file, protocol=pickle_protocol)
        save_path.chmod(0o664)

    def _load_from_dataframe(
        self,
        df: gpd.GeoDataFrame,
        tolerance: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """
        Convert geopandas dataframe to networkx graph.

        This code takes around 3 minutes to run on JASMIN for the Chernobyl
        habitat data.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing polygon objects from
                a shape file.
            tolerance (float, optional): Adds edges between neighbours that are
                at most `tolerance` units apart. Defaults to 0.

        Raises:
            ValueError: If `tolerance` < 0, if `class_label` or `geometry` are
                not columns in the dataframe.

        Returns:
            gpd.GeoDataFrame: The dataframe containing polygon objects.
        """
        if tolerance < 0.0:
            raise ValueError("`tolerance` must be greater than 0.")
        if self._columns_to_rename is not None:
            df = df.rename(columns=self._columns_to_rename)
        if "class_label" not in df.columns:
            raise ValueError("`class_label` must be a column in the dataframe.")
        if "geometry" not in df.columns:
            raise ValueError("`geometry` must be a column in the dataframe.")
        # Assign crs
        if df.crs is None:
            df.crs = self._crs
        if self._crs is not None:
            df = df.to_crs(self._crs)

        # Reset index to ensure consistent indices
        df = df.reset_index(drop=True)
        # Using this list and iterating through it is slightly faster than
        # iterating through df due to the dataframe overhead
        geom: List[shapely.Polygon] = df["geometry"].tolist()
        # this dict maps polygon row numbers in df to a list
        # of neighbouring polygon row numbers
        graph_dict = {}

        if tolerance > 0:
            # Expand the borders of the polygons by `tolerance```
            new_polygons: List[shapely.Polygon] = (
                df["geometry"].buffer(tolerance).tolist()
            )
        # Creating nodes (=vertices) and finding neighbors
        for index, polygon in tqdm(
            enumerate(geom),
            desc="Step 1 of 2: Creating nodes and finding neighbours",
            total=len(geom),
        ):
            if tolerance > 0:
                # find the indexes of all polygons which intersect with this one
                neighbours = df.sindex.query(
                    new_polygons[index], predicate="intersects"
                )
            else:
                neighbours = df.sindex.query(polygon, predicate="intersects")

            graph_dict[index] = neighbours
            # add each polygon as a node to the graph with useful attributes
            self.graph.add_node(
                index,
                rep_point=polygon.representative_point(),
                area=polygon.area,
                perimeter=polygon.length,
                bounds=polygon.bounds,
            )
        # iterate through the dict and add edges between neighbouring polygons
        for polygon_id, neighbours in tqdm(
            graph_dict.items(), desc="Step 2 of 2: Adding edges"
        ):
            for neighbour_id in neighbours:
                if polygon_id != neighbour_id:
                    self.graph.add_edge(polygon_id, neighbour_id)

        # add index name
        df.index.name = "node_index"
        return df

    def merge_nodes(
        self,
        node_list: List[int],
        class_label: Union[int, str],
        final_index: int = None,
    ) -> None:
        """
        Merge a list of nodes in the graph together into a single node.

        This will create a node with a neighbour list and polygon which is the
        union of the nodes in `node_list`.

        Args:
            node_list (List[int]): List of integer node indexes in the graph.
            class_label (int or str): Class label for the resulting node.
            final_index (int, optional): Index to assign to the resulting node.
                Defaults to None, in which case it becomes the highest valid index
                in the dataframe + 1.

        Raises:
            ValueError: If `final_index` is an existing node not in `node_list`,
            or if `node_list` does not contain any existing nodes.
        """
        node_list = [node for node in node_list if self.graph.has_node(node)]
        if len(node_list) == 0:
            raise ValueError("`node_list` must contain at least one node in the graph.")
        if final_index is None:
            final_index = self.df.last_valid_index() + 1
        elif final_index not in node_list and self.graph.has_node(final_index):
            raise ValueError(
                "`final_index` cannot be an existing node that is not in `node_list`."
            )

        # Build set of all neighbours of nodes in node_list, excluding the
        # nodes in node_list.
        adjacency_set = set()
        for node in node_list:
            adjacency_set.update(list(self.graph.neighbors(node)))
        adjacency_set -= adjacency_set.intersection(node_list)

        # Build union polygon.
        polygon = self.df["geometry"].loc[node_list].unary_union

        # Remove nodes from graph and rows from df
        self._remove_nodes(node_list)
        # Add final node to graph and df
        self._add_node(
            final_index, adjacency_set, geometry=polygon, class_label=class_label
        )

    def merge_classes(
        self, class_list: List[Union[str, int]], new_name: Union[str, int]
    ) -> None:
        """
        Merge multiple classes together into one by renaming the class labels.

        Args:
            new_name (Union[str, int]): The new name for the combined class,
                either a string or an int.
            class_list (List): The list of names of class labels to combine.
                Every name in the list must be in the GeoGraph.

        Raises:
            ValueError: If `class_list` contains a class name not already in
                the GeoGraph.
        """
        if not set(class_list).issubset(self.df["class_label"].unique()):
            raise ValueError("`class_list` must only contain valid class names.")
        self.df.loc[self.df["class_label"].isin(class_list), "class_label"] = new_name

    def add_habitat(
        self,
        name: str,
        valid_classes: List[Union[str, int]],
        barrier_classes: Optional[List[Union[str, int]]] = None,
        max_travel_distance: float = 0.0,
        add_distance: bool = False,
        add_component_edges: bool = False,
    ) -> None:
        """
        Create HabitatGeoGraph object and store it in habitats dictionary.

        Creates a habitat subgraph of the main GeoGraph that only contains edges
        between nodes in `valid_classes` as long as they are less than
        `max_travel_distance` apart. All nodes which are not in `valid_classes`
        are not in the resulting habitat graph. This graph is then stored as its
        own HabitatGeoGraph object with all meta information.

        Args:
            name (str): The name of the habitat.
            valid_classes (List): A list of class labels which make up the habitat.
            barrier_classes (List): Defaults to None.
            max_travel_distance (float): The maximum distance the animal(s) in
                the habitat can travel through non-habitat areas. The habitat graph
                will contain edges between any two nodes that have a class label in
                `valid_classes`, as long as they are less than `max_travel_distance`
                units apart. Defaults to 0, which will only create edges between
                directly neighbouring areas.
            add_distance (bool, optional): Whether or not to add the distance
                between polygons as an edge attribute in the habitat graph. Defaults
                to False.
            add_component_edges (bool, optional): Whether to add edges between
                nodes in the ComponentGeoGraph (which is automatically created as an
                attribute of the resulting HabitatGeoGraph) with edge weights that
                are the distance between neighbouring components. Can be
                computationally expensive. Defaults to False.

        Raises:
            ValueError: If max_travel_distance < 0.
        """
        if max_travel_distance < 0.0:
            raise ValueError("`max_travel_distance` must be greater than 0.")
        if barrier_classes is None:
            barrier_classes = []
        hgraph: nx.Graph = deepcopy(self.graph)
        # Remove all edges in the graph, then at the end we only have edges
        # between nodes less than `max_travel_distance` apart
        hgraph.clear_edges()
        # Get dict to convert between iloc indexes and loc indexes
        # These are different only if nodes have been removed from the df
        idx_dict: Dict[int, int] = dict(zip(range(len(self.df)), self.df.index.values))
        # Get dicts of polygons and buff polygons to avoid repeatedly querying
        # the dataframe. These dicts accept loc indexes
        polygons: Dict[int, shapely.Polygon] = self.df["geometry"].to_dict()
        if max_travel_distance > 0:
            # Vectorised buffer on the entire df to calculate the expanded polygons
            # used to get intersections.
            buff_polygons = self.df["geometry"].buffer(max_travel_distance).to_dict()
        # Remove non-habitat nodes from habitat graph
        # np.where is very fast here and gets the iloc based indexes
        # Combining it with the set comprehension reduces time by an order of
        # magnitude compared to `set(self.df.loc[])`
        valid_class_indices = np.isin(self.class_label, valid_classes)
        invalid_idx = {idx_dict[i] for i in np.where(~valid_class_indices)[0]}
        hgraph.remove_nodes_from(invalid_idx)

        for node in tqdm(
            hgraph.nodes, desc="Generating habitat graph", total=len(hgraph)
        ):
            polygon = polygons[node]
            if max_travel_distance > 0:
                buff_poly_bounds = buff_polygons[node].bounds
                buff_poly = prep(buff_polygons[node])
            else:
                buff_poly_bounds = polygon.bounds
                buff_poly = prep(polygon)
            # Query rtree for all polygons within `max_travel_distance` of the original
            for nbr in self.rtree.intersection(buff_poly_bounds):
                # Necessary to correct for the rtree returning iloc indexes
                nbr = idx_dict[nbr]
                # If a node is not a habitat class node, don't add the edge
                if nbr == node or nbr in invalid_idx:
                    continue
                # Otherwise add the edge with distance attribute
                nbr_polygon = polygons[nbr]
                if not hgraph.has_edge(node, nbr) and buff_poly.intersects(nbr_polygon):
                    if add_distance:
                        if max_travel_distance == 0:
                            dist = 0.0
                        else:
                            dist = polygon.distance(nbr_polygon)
                        hgraph.add_edge(node, nbr, distance=dist)
                    else:
                        hgraph.add_edge(node, nbr)
        # Add habitat to habitats dict
        habitat = HabitatGeoGraph(
            data=self.df.iloc[np.where(valid_class_indices)[0]],
            name=name,
            graph=hgraph,
            valid_classes=valid_classes,
            barrier_classes=barrier_classes,
            max_travel_distance=max_travel_distance,
            add_distance=add_distance,
            add_component_edges=add_component_edges,
        )
        self.habitats[name] = habitat

    def apply_to_habitats(self, func: Callable, **kwargs) -> List[Any]:
        """
        Apply a function to all habitats in this GeoGraph.

        The function must be a method of GeoGraph or HabitatGeoGraph. **kwargs
        are applied to the function - all passed arguments must be specified
        with the keyword.

        Args:
            func (Callable): A function that is a method of GeoGraph or
                HabitatGeoGraph. This must not be a method of an instance of
            GeoGraph; it can only be an actual method definition, such as
                `GeoGraph.merge_nodes`.

        Raises:
            ValueError: If `func` is not a method of GeoGraph or HabitatGeoGraph.

        Returns:
            List[Any]: A list of the returned results.
        """
        # __qualname__ should be something like "GeoGraph.merge_nodes"
        if (
            not func.__qualname__.split(".")[0] in ("GeoGraph", "HabitatGeoGraph")
            or "self" not in inspect.signature(func).parameters
        ):
            raise ValueError("`func` is not a method of GeoGraph or HabitatGeoGraph.")
        return [func(self=habitat, **kwargs) for habitat in self.habitats.values()]

    def get_graph_components(
        self, calc_polygons: bool = True, add_distance_edges: bool = False
    ) -> ComponentGeoGraph:
        """Return a GeoGraph with the connected components of this graph.

        This method determines the individual disconnected graph components that
        make up the graph of the GeoGraph object. It returns a ComponentGeoGraph
        object such that each row of the GeoDataFrame and each node in the
        networkx.Graph correspond to a connected component in the main graph,
        and the polygons in the dataframe are the union of all individual
        polygons making up a component in the main graph.

        Warning: this method is very slow if `calc_polygons=True` and the graph
        consists mostly of one big component, since taking the union is expensive.

        This method allows for the UI to visualise components and output their
        number as a metric.

        More info on the definition of graph components can be found here:
        https://en.wikipedia.org/wiki/Component_(graph_theory)

        Args:
            calc_polygons (bool, optional): This determines whether to calculate
                the polygons which are the union of all the polygons that make up
                each component, and load a `ComponentGeoGraph` with a corresponding
                dataframe containing these components. This can be time consuming
                if there is a very large component. Defaults to True.
            add_distance_edges (bool, optional): This determines whether to add
                edges between every pair of nodes, with the distance between their
                corresponding polygons as an edge attribute. Defaults to False.

        Returns:
            ComponentGeoGraph: A ComponentGeoGraph containing the resulting
                GeoDataFrame (if `calc_polygons=True`) and the list of graph
                components.
        """
        components: List[set] = list(nx.connected_components(self.graph))
        if calc_polygons:
            geom = [self.df["geometry"].loc[comp].unary_union for comp in components]
            gdf = gpd.GeoDataFrame(
                {"geometry": geom, "class_label": -1}, crs=self.df.crs
            )
            comp_geograph = ComponentGeoGraph(
                components, gdf, add_distance_edges=add_distance_edges
            )
        else:
            comp_geograph = ComponentGeoGraph(components)
        if not hasattr(self, "components"):
            self.components = comp_geograph
        return comp_geograph

    def get_metric(
        self, name: str, class_value: Optional[int] = None, **metric_kwargs
    ) -> metrics.Metric:
        """
        Calculate and save the metric with name `name` for the current GeoGraph.

        Args:
            name (str): The name of a valid metric for a GeoGraph.
            class_value(int): The landcover class label if a class level metric is
                desired. None if a landscape/component level metric is desired.
                Defaults to None.

        Returns:
            metrics.Metric: The Metric object, containing the value as well as
                other information about the metric.
        """
        # Case 1: Landscape/component level metrics
        if class_value is None:
            try:
                result = self.metrics[name]
            except KeyError:
                # pylint: disable=protected-access
                result = metrics._get_metric(name=name, geo_graph=self, **metric_kwargs)
                self.metrics[name] = result
            return result

        # Case 2: Class level metrics
        else:
            try:
                result = self.class_metrics[name][class_value]
            except KeyError:
                # pylint: disable=protected-access
                result = metrics._get_metric(
                    name=name, geo_graph=self, class_value=class_value, **metric_kwargs
                )
                if name in self.class_metrics.keys():
                    self.class_metrics[name][class_value] = result
                else:
                    self.class_metrics[name] = {class_value: result}
            return result

    def get_class_metrics(
        self,
        names: Optional[Union[str, Sequence[str]]] = None,
        classes: Optional[Union[str, Sequence[int], np.ndarray]] = None,
        **metric_kwargs,
    ) -> pd.DataFrame:
        """
        Return class-level metrics for the landcover classes in the given GeoGraph.

        If arguments are omitted, all class level metrics are calculated.

        Args:
            names (Optional[Union[str, Sequence[str]]], optional): Names of the metrics
                to calculate. If None, then all available class metrics are calculated
                for the given classses. Defaults to None.
            classes (Optional[Union[str, Sequence[int]]], optional): Class labels of
                the classes to calculate. If None, then the given metrics are calculated
                for all classes. Defaults to None.
            **metric_kwargs: Any kwargs that should be passed on to the metrics.

        Returns:
            pd.DataFrame: A dataframe containing the metrics for the selected classes
        """
        # Convert to iterable if single values are given
        if names is None:
            names = list(CLASS_METRICS_DICT.keys())
        elif isinstance(names, str):
            names = [names]
        if classes is None:
            classes = self.classes
        elif isinstance(classes, int):
            classes = [classes]

        # Create metrics id not yet present
        result: Dict[str, List] = {name: [] for name in names}
        for name in names:
            for class_value in classes:
                result[name].append(
                    self.get_metric(
                        name, class_value=class_value, **metric_kwargs
                    ).value
                )

        return pd.DataFrame(result, index=classes)

    def get_patch_metrics(self) -> pd.DataFrame:
        """
        Return patch-level metrics and append them to `self.df`.

        Calculates "area", "perimeter", "perimeter_area_ratio", "shape_index" and
        "fractal_dimension" for each patch

        Returns:
            pd.DataFrame: Dataframe containing the patch level metrics.
        """
        self.df["area"] = self.df.area
        self.df["perimeter"] = self.df.length
        self.df["perimeter_area_ratio"] = self.df["perimeter"] / self.df["area"]
        self.df["shape_index"] = 0.25 * self.df["perimeter"] / np.sqrt(self.df["area"])
        self.df["fractal_dimension"] = (
            2 * np.log(0.25 * self.df["perimeter"]) / np.log(self.df["area"])
        )

        return self.df[
            [
                "class_label",
                "area",
                "perimeter",
                "perimeter_area_ratio",
                "shape_index",
                "fractal_dimension",
            ]
        ]

    def identify_node(
        self, node_id: int, other_graph: GeoGraph, mode: str
    ) -> List[int]:
        """Return all node ids in `other_graph` which identify with `node_id`."""
        return binary_graph_operations.identify_node(
            self.df.loc[node_id], other_graph=other_graph, mode=mode
        )

    def _remove_node(self, node_id: int):
        self._remove_nodes([node_id])

    def _remove_nodes(self, node_ids: Iterable[int]):
        # Remove node from graph (automatically removes edges)
        self.graph.remove_nodes_from(node_ids)
        # Remove data of node from df
        self.df = self.df.drop(index=node_ids)

    def _add_node(
        self,
        node_id: int,
        adjacencies: Iterable[int],
        requires_sorting: bool = True,
        **data,
    ):
        """
        Add node to graph.

        Args:
            node_id (int): The id of the node to add.
                adjacencies (Iterable[int]): Iterable of node ids which are adjacent
                to the new node.
            requires_sorting (bool, optional): Whether or not to sort the dataframe
                index after adding the node. Defaults to True.
        """
        # Collect all data on node in one dict
        node_data = dict(data.items())

        # Add node to graph
        self.graph.add_node(
            node_id,
            rep_point=node_data["geometry"].representative_point(),
            area=node_data["geometry"].area,
            perimeter=node_data["geometry"].length,
            class_label=node_data["class_label"],
            bounds=node_data["geometry"].bounds,
        )

        # Add node data to dataframe
        missing_cols = {
            key: None for key in set(self.df.columns) - set(node_data.keys())
        }

        self.df.loc[node_id] = gpd.GeoSeries({**data, **missing_cols}, crs=self.df.crs)
        if requires_sorting:
            self.df = self.df.sort_index()

        # Add edges to adjacency list
        self.graph.add_edges_from(
            zip_longest([node_id], adjacencies, fillvalue=node_id)
        )

    # pylint: disable=dangerous-default-value
    def _add_nodes(self, node_ids, adjacencies, node_data={}, **data):
        raise NotImplementedError

    def _merge_graph(self, other: GeoGraph) -> GeoGraph:

        for node_id in self.rtree.intersection(other.bounds):
            print(self.identify_node(node_id, other, mode="edge"))

        raise NotImplementedError


class HabitatGeoGraph(GeoGraph):
    """Class to represent a habitat GeoGraph."""

    def __init__(
        self,
        data: Union[gpd.GeoDataFrame, str, os.PathLike],
        name: Optional[str] = None,
        graph: Optional[nx.Graph] = None,
        valid_classes: Optional[List[Union[str, int]]] = None,
        barrier_classes: Optional[List[Union[str, int]]] = None,
        max_travel_distance: Optional[float] = 0,
        add_distance: bool = False,
        add_component_edges: bool = False,
    ) -> None:
        """
        Class to represent a habitat GeoGraph.

        This class can load a habitat GeoGraph from a GeoDataFrame and networkx
        graph object, or alternatively load saved pickle or compressed pickle
        file with the graph, dataframe, and all metadata. Valid saved file formats
        are .pickle, .pkl, .gz, or .bz2.

        Args:
            data: (GeoDataFrame or Path): Either a dataframe with the polygon
                data for the habitat graph nodes, or a path to a saved habitat. If
                it is a GeoDataFrame, then the other arguments in this init are
                mandatory (except for `add_distance` and `add_component_edges`)
            name (str, optional): The name of the habitat.
            graph (nx.Graph, optional): A networkx graph representing the habitat.
                Defaults to None.
            valid_classes (List, optional): A list of class labels which make up
                the habitat.
            barrier_classes (List, optional): A list of barrier class labels.
            max_travel_distance (float, optional): The maximum distance the
                animal(s) in the habitat can travel through non-habitat areas. The
                habitat graph will contain edges between any two nodes that have a
                class label in `valid_classes`, as long as they are less than
                `max_travel_distance` units apart.
            add_distance (bool, optional): Whether or not to add the distance
                between polygons has been added as an edge attribute in `graph`.
            add_component_edges (bool, optional): Whether to add edges between
                nodes in the ComponentGeoGraph created automatically for this
                habitat with edge weights that are the distance between neighbouring
                components. Can be computationally expensive. Defaults to False.

        Raises:
            ValueError: If `data` is of an unknown type, or if `data` is a file
            path of an invalid suffix.
        """
        # pylint: disable=super-init-not-called
        if isinstance(data, gpd.GeoDataFrame):
            # Note that index is not reset, so it contains the loc indices of the
            # same patches in the main df
            self.df: gpd.GeoDataFrame = data
            if (
                name is not None
                and graph is not None
                and valid_classes is not None
                and barrier_classes is not None
                and max_travel_distance is not None
            ):
                self.name: str = name
                self.graph: nx.Graph = graph
                self.valid_classes: List[Union[str, int]] = valid_classes
                self.barrier_classes: List[Union[str, int]] = barrier_classes
                self.max_travel_distance: float = max_travel_distance
                self.add_distance: bool = add_distance

        elif isinstance(data, (str, os.PathLike)):
            load_path = pathlib.Path(data)
            assert load_path.exists()
            # Load from saved graph
            if load_path.suffix in (".pickle", ".pkl", ".gz", ".bz2"):
                self._load_from_graph_path(load_path)
            else:
                raise ValueError(
                    f"""Extension {load_path.suffix} unknown.
                                 Must be one of .pickle, .pkl, .gz, .bz2."""
                )
        else:
            raise ValueError(
                """Type of `data` unknown. Must be a dataframe or file path."""
            )
        self.metrics: Dict[str, Metric] = {}
        self.class_metrics: Dict[str, Dict[Union[str, int], Metric]] = {}
        print("Calculating components...")
        self.components = self.get_graph_components(
            calc_polygons=True, add_distance_edges=add_component_edges
        )

        print(
            f"Habitat successfully loaded with {self.graph.number_of_nodes()} nodes",
            f"and {self.graph.number_of_edges()} edges.",
        )

    def _load_from_graph_path(self, load_path: pathlib.Path) -> None:
        """
        Load networkx graph and dataframe objects from a pickle file.

        Args:
            load_path (pathlib.Path): Path to a pickle file. Can be compressed
            with gzip or bz2.

        Returns:
            gpd.GeoDataFrame: The dataframe containing polygon objects.
        """
        if load_path.suffix == ".bz2":
            with bz2.BZ2File(load_path, "rb") as bz2_file:
                data = pickle.load(bz2_file)
        elif load_path.suffix == ".gz":
            with gzip.GzipFile(load_path, "rb") as gz_file:
                data = pickle.loads(gz_file.read())
        else:
            with open(load_path, "rb") as file:
                data = pickle.load(file)
        self.df = data["dataframe"]
        self.name = data["name"]
        self.valid_classes = data["valid_classes"]
        self.barrier_classes = data["barrier_classes"]
        self.max_travel_distance = data["max_travel_distance"]
        self.add_distance = data["add_distance"]

    def save_habitat(self, save_path: pathlib.Path) -> None:
        """
        Save graph with attributes and dataframe as pickle file. Can be compressed.

        Args:
            save_path (pathlib.Path): Path to a pickle file. Can be compressed
            with gzip or bz2 by passing filenames ending in `gz` or `bz2`.

        Raises:
            ValueError: If `save_path` is not a pickle, gz, or bz2 file.
        """
        if save_path.suffix not in (".pickle", ".pkl", ".gz", ".bz2"):
            raise ValueError(
                """Argument `save_path` should be a pickle file or
                compressed file."""
            )
        data = {
            "graph": self.graph,
            "dataframe": self.df,
            "name": self.name,
            "valid_classes": self.valid_classes,
            "barrier_classes": self.barrier_classes,
            "max_travel_distance": self.max_travel_distance,
            "add_distance": self.add_distance,
        }
        if save_path.suffix == ".bz2":
            with bz2.BZ2File(save_path, "wb") as bz2_file:
                pickle.dump(data, bz2_file)
        elif save_path.suffix == ".gz":
            with gzip.GzipFile(save_path, "wb") as gz_file:
                gz_file.write(pickle.dumps(data))
        else:
            with open(save_path, "wb") as file:
                pickle.dump(data, file)
        save_path.chmod(0o664)


class ComponentGeoGraph(GeoGraph):
    """Class to represent the connected components of a GeoGraph."""

    def __init__(
        self,
        components_list: List[set],
        df: Optional[gpd.GeoDataFrame] = None,
        add_distance_edges: bool = False,
    ) -> None:
        """
        Class for the connected components of a GeoGraph.

        This class can load a graph from only a list of components, in which
        case it will not have a dataframe and the `has_df` class attribute will
        be False, or it can load from both a list of components and a dataframe.
        In the latter case, edges with the polygon distance can be added between
        each pair of nodes (where each node corresponds to a connected component
        in the original graph).

        WARNING: using `add_distance_edges=True` is incredibly slow for anything
        other than the smallest habitats.

        Args:
            components_list (List[set]): A list of sets, where each set contains
                the node indices making up a component from the original graph.
            df (Optional[gpd.GeoDataFrame], optional): A GeoDataFrame in which
                each row contains a polygon that represents a component from a
                GeoGraph. This is optional, and if not passed the graph will be
                created with no edges. Defaults to None.
            add_distance_edges (bool, optional): Boolean that determines whether
                to add edges between every pair of nodes, with the distance between
                their corresponding polygons as an edge attribute. Defaults to False.
        """
        # pylint: disable=super-init-not-called
        self.has_df: bool = True
        self.graph: nx.Graph = nx.Graph()
        self.components_list: List[set] = components_list
        if df is not None:
            self.has_distance_edges: bool = add_distance_edges
            self.df = self._load_from_dataframe(df)
        else:
            self.has_df = False
            self.has_distance_edges = False
            self.graph = nx.empty_graph(len(self.components_list))
        self.metrics: Dict[str, Metric] = {}

    def _load_from_dataframe(
        self,
        df: gpd.GeoDataFrame,
        tolerance: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """
        Load graph from dataframe.

        If `self.has_distance_edges` is True, then this will load a complete graph
        and add the distance between each pair of polygons as an edge attribute.
        Otherwise, it will load an empty graph.

        In both cases, information about the polygons will be stored as node
        attributes.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing polygon objects.
            tolerance (float, optional): Redundant argument to ensure interface
                is consistent with parent class. Defaults to 0.0.

        Returns:
            gpd.GeoDataFrame: A processed GeoDataFrame containing polygon objects.
        """
        # Reset index to ensure consistent indices
        df = df.reset_index(drop=True)
        # Using this list and iterating through it is slightly faster than
        # iterating through df due to the dataframe overhead
        geom: List[shapely.Polygon] = df["geometry"].tolist()
        if self.has_distance_edges:
            self.graph = nx.complete_graph(len(df))
        else:
            self.graph = nx.empty_graph(len(df))
        # Add node attributes
        for node in tqdm(
            self.graph.nodes, desc="Constructing graph", total=len(self.graph)
        ):
            polygon = geom[node]
            self.graph.nodes[node]["rep_point"] = polygon.representative_point()
            self.graph.nodes[node]["area"] = polygon.area
            self.graph.nodes[node]["perimeter"] = polygon.length
            self.graph.nodes[node]["bounds"] = polygon.bounds

        # Add edge attributes if necessary
        if self.has_distance_edges:
            for u, v, attrs in tqdm(
                self.graph.edges.data(data=True),
                desc="Calculating edge weights",
                total=len(self.graph.edges),
            ):
                attrs["distance"] = geom[u].distance(geom[v])
        # add index name
        df.index.name = "node_index"
        return df
