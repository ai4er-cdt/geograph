"""Module for analysing multiple GeoGraph objects."""
from __future__ import annotations

import datetime
from bisect import bisect_left
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.binary_graph_operations import NodeMap, identify_graphs
from src.geograph import GeoGraph

# type alias
TimeStamp = Union[int, datetime.datetime]


class NotCachedError(Exception):
    """Basic exception for values which were not yet cached."""


class TimedGeoGraph(GeoGraph):
    """Wrapper class for GeoGraphs with a time attribute"""

    def __init__(self, time: TimeStamp, **geographargs) -> None:
        """
        Simple wrapper class for GeoGraphs with a time attribute.

        Args:
            time (TimeStamp): The timestamp of a given Geograph. Must be an integer or
                a python datetime object.
            **geographargs: Any argument to the GeoGraph class
        """
        super().__init__(**geographargs)
        self._time = time

    @property
    def time(self) -> TimeStamp:
        """Return the time attribute."""
        return self._time


class GeoGraphTimeline:
    """Timeline of multiple GeoGraphs."""

    def __init__(
        self, data: Union[List[TimedGeoGraph], Dict[TimeStamp, GeoGraph]]
    ) -> None:
        """
        Creates a timeline of multiple GeoGraphs, for time-series and change-detection
        analyses.

        The `data` must be a list of TimedGeoGraph objects or a dictionary where the
        keys correspond to the respective time-stamps of each GeoGraph. The GeoGraphs
        in `data` will be sorted from earliest to latest timestamp and added to the
        GeoGraphTimeline.

        Landscape, Habitat-component, Class-value and patch level time-series analyses
        are supported.

        Args:
            data (Union[List[TimedGeoGraph], Dict[TimeStamp, GeoGraph]]): A list of
                TimedGeoGraphs or a dictionary where keys correspond to times and values
                to GeoGraph objects of the ecosystem at the specified time.

        Raises:
            NotImplementedError: For any other `data` argument. (In the future we will
                add funcitonality to save timelines and load from disk.)
        """

        # Initialize empty graphs dict
        self._graphs: Dict[TimeStamp, GeoGraph] = dict()
        self.habitats: Dict[str, GeoGraphTimeline] = {}
        # Fill the graphs dict with graphs from `data`
        if isinstance(data, list):
            self._load_from_sequence(graph_list=data)
        elif isinstance(data, dict):
            self._load_from_dict(graph_dict=data)
        else:
            raise NotImplementedError

        # Initialize empty node map cache dictionary
        self._node_map_cache: Dict[(TimeStamp, TimeStamp), NodeMap] = dict()

    @property
    def times(self) -> List[TimeStamp]:
        """Return list of valid time stamps for this GeoGraphTimeline"""
        return list(self._graphs.keys())

    @property
    def graphs(self) -> Dict[TimeStamp, GeoGraph]:
        """Return sorted list of GeoGraphs in this GeoGraphTimeline"""
        return self._graphs

    def __getitem__(self, time: TimeStamp) -> GeoGraph:
        """Return the graph at this given time stamp in the GeoGraphTimeline"""
        try:
            return self._graphs[time]
        except KeyError as error:
            raise KeyError(f"Graph for time index `{time}` does not exist.") from error

    def __len__(self) -> int:
        """Return the number of timestamps in the GeoGraphTimeline"""
        return len(self._graphs)

    def __iter__(self) -> GeoGraph:
        """Iterate over graphs in the GeoGraphTimeline in order (earlier to later)"""
        return iter(self._graphs.values())

    def _sort_by_time(self, reverse: bool = False) -> None:
        """
        Sort the graphs in GeoGraphTimeline accroding to timestamp from earlier to later

        Args:
            reverse (bool, optional): If False, start with the earliest date.
                If True, sort starting with the latest date. Defaults to False.
        """
        self._graphs = {
            time: self._graphs[time] for time in sorted(self._graphs, reverse=reverse)
        }

    def _load_from_sequence(self, graph_list: List[TimedGeoGraph]) -> None:
        """Loads the sorted list of timed geographs into the timeline. """

        # Make sure list is sorted in ascending time order (earliest = left)
        by_time = lambda item: item.time
        self._graphs = {graph.time: graph for graph in sorted(graph_list, key=by_time)}

    def _load_from_dict(self, graph_dict: Dict[TimeStamp, GeoGraph]) -> None:
        """Loads the dictionary of graphs into the timeline and sorts them by time"""

        self._graphs = graph_dict
        self._sort_by_time()

    def identify_graphs(
        self, time1: TimeStamp, time2: TimeStamp, use_cached: bool = True
    ) -> NodeMap:
        """
        Identify the nodes between the graph at time `time1` and `time2` in the timeline

        Args:
            time1 (TimeStamp): timestamp index of the first graph (will be src_graph)
            time2 (TimeStamp): timestamp index of the second graph (will be trg_graph)
            use_cached (bool, optional): Iff True, use cached NodeMaps from previous
                computations. Defaults to True.

        Returns:
            NodeMap: The one-to-many node mapping from `self[time1]` to `self[time2]`
        """

        if use_cached:
            try:
                return self.node_map_cache(time1, time2)
            except NotCachedError:
                pass

        self._node_map_cache[(time1, time2)] = identify_graphs(
            self[time1], self[time2], mode="interior"
        )

        return self._node_map_cache[(time1, time2)]

    def node_map_cache(self, time1: TimeStamp, time2: TimeStamp) -> NodeMap:
        """
        Return cached NodeMap from the graph at `time1` to that at `time2`.

        Args:
            time1 (TimeStamp): Time stamp of the first graph (src_graph)
            time2 (TimeStamp): Time stamp of the second graph (trg_graph)

        Raises:
            NotCachedError: If the combination (time1, time2) or its inverse
                (time2, time1) have not been cached yet.

        Returns:
            NodeMap: The NodeMap to identify nodes from `self[time1]` with `self[time2]`
        """
        if (time1, time2) in self._node_map_cache.keys():
            return self._node_map_cache[(time1, time2)]
        elif (time2, time1) in self._node_map_cache.keys():
            map_from_inverse = self._node_map_cache[(time2, time1)].invert()
            self._node_map_cache[(time1, time2)] = map_from_inverse
            return map_from_inverse
        else:
            raise NotCachedError

    def _empty_node_map_cache(self) -> None:
        """ Empties the node map cache."""
        self._node_map_cache = dict()

    def timestack(self, use_cached: bool = True) -> List[NodeMap]:
        """
        Performs node identification between adjacent time-slices in the graph.

        Args:
            use_cached (bool, optional): If True, reuses prior node-identification
                computations. Defaults to True.

        Returns:
            List[NodeMap]: An ordered list of the of the node maps between each two
                adjacent time-slices in the GeoGraphTimeline
        """
        node_maps = []
        for time1, time2 in zip(self.times, self.times[1:]):
            node_maps.append(self.identify_graphs(time1, time2, use_cached))
        return node_maps

    def timediff(self, use_cached: bool = True):
        raise NotImplementedError

    def node_diff_cache(self, time1: TimeStamp, time2: TimeStamp):
        raise NotImplementedError

    def get_metric(self, name: str, class_value: Optional[int] = None) -> xr.DataArray:
        """
        Return the time-series for the given metric.

        For class-level metrics pass a class_value argument. For landscape/component
        level metrics omit the class_value argument.

        Args:
            name (str): Name of the metric to compute
            class_value (Optional[int], optional): Provide a class value if you wish
            to calculate a class-level metric. Leave as None for calculating
            landscape/habitat level metrics. Defaults to None.

        Returns:
            xr.DataArray: A DataArray containing the metric time series for the graphs
                in the given GeoGraphTimeline
        """

        # Calculate metrics
        metrics = [
            graph.get_metric(name=name, class_value=class_value) for graph in self
        ]

        # Set up metadata
        attrs = {"description": metrics[0].description, "unit": metrics[0].unit}
        if class_value:
            attrs["class_label"] = class_value

        metric_timeseries = xr.DataArray(
            [metric.value for metric in metrics],
            dims=["time"],
            coords=[self.times],
            name=name,
            attrs=attrs,
        )

        return metric_timeseries

    def get_class_metrics(
        self,
        names: Optional[Union[str, Iterable[str]]] = None,
        class_values: Optional[Union[int, Iterable[int]]] = None,
    ) -> xr.DataArray:
        """
        Return the time-series of the selected class metrics for the given `classes`.


        Args:
            names (Optional[Union[str, Iterable[str]]], optional): Names of the
                class-level metrics to calculate. If None, all available class metrics
                are calculated. Defaults to None.
            class_values (Optional[Union[int, Iterable[int]]], optional): Class values
                for the classes for which the metrics should be calculated. If None, the
                metrics are calcluated for all available classes in the
                GeoGraphTimeline. Classes which do not exist a certain point in time
                will have `np.nan` values. Defaults to None.

        Returns:
            xr.DataArray: A three dimensional data array containing the time-series
                class level metrics for the selected classes with dimensions
                (time, class_label, metric).
        """

        class_metric_dfs = [
            xr.DataArray(
                graph.get_class_metrics(names, class_values),
                dims=["class_label", "metric"],
            )
            for graph in self
        ]

        return xr.concat(class_metric_dfs, dim=pd.Index(self.times, name="time"))

    def get_patch_metrics(
        self, aggregator: Union[str, Callable] = "mean"
    ) -> xr.DataArray:
        """
        Return aggregated patch distribution metrics for all classes.

        Args:
            aggregator (Union[str, Callable], optional): Aggregation function to use
            on the patch-level statistics. Defaults to "mean".

        Returns:
            xr.DataArray: A three dimensional array containing the time-series of the
                aggregated patch-level distributions for each class. Dimension are
                (time, class_label, metric)
        """

        metrics_dfs = []
        for graph in self:
            patch_metrics = (
                graph.get_patch_metrics().groupby("class_label").aggregate(aggregator)
            )

            metrics_dfs.append(
                xr.DataArray(
                    patch_metrics,
                    dims=["class_label", "metric"],
                )
            )

        return xr.concat(metrics_dfs, dim=pd.Index(self.times, name="time"))

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
        Create HabitatGeoGraph for each graph in the timeline.

        Creates a habitat subgraph for each of the main GeoGraph objects in the timeline
        that only contains edges between nodes in `valid_classes` as long as they are
        less than `max_travel_distance` apart.
        All nodes which are not in `valid_classes` are not in the resulting habitat
        graph. This graph is then stored as its own HabitatGeoGraph object with all
        meta information.

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

        for graph in self:
            graph.add_habitat(
                name=name,
                valid_classes=valid_classes,
                barrier_classes=barrier_classes,
                max_travel_distance=max_travel_distance,
                add_distance=add_distance,
                add_component_edges=add_component_edges,
            )

        self.habitats[name] = GeoGraphTimeline(
            {year: graph.habitats[name] for year, graph in self.graphs.items()}
        )

    def calculate_node_dynamics(self, time: TimeStamp) -> pd.Series:
        """
        Classify the dynamic behavior of each node of the graph at the given `time`.

        The node dynamics reflects the type of change that a node has undergone between
        two adjacent time-slices of a GeoGraphTimeline.
        For each node at the selected `time` the classified node dynamics is one of:
            `birth`: The child node has no ancestors in `nodemap` and newly appeared.
            `split`: The child node was created as a split off of an ancestral node.
            `unchanged`: The child node has one ancestor with the same characteristics.
            `grew`: The child node has one ancestor and has increased in area.
            `shrank`: The child node has one ancestor and has decreased in area.
            `complex`: The child node has numerous ancestors and siblings
            `merged`: The child node has several ancestors and no siblings. It was
                created from a merge of several ancestral nodes.

        Args:
            time (TimeStamp): The time in GeoGraphTimeline for which the node dynamics
                should be classified. Must be one of the times in `self.times`.

        Raises:
            UserWarning: If an inexistent timeslice is accessed.

        Returns:
            pd.Series: A pandas series with the node dynamics for each node
                the graph at the given time (self[time]).
        """

        if "node_dynamic" in self[time].df.columns:
            return self[time].df["node_dynamic"]
        else:
            # Determine index of the selected time in self.times
            time_index = bisect_left(self.times, time)

            if time_index == 0:
                raise UserWarning(
                    "Cannot calculate node dyamics for first graph in timeline."
                )

            prior_time = self.times[time_index - 1]
            node_map = self.node_map_cache(prior_time, time)

            # Helper function to calculate dynamics type via a single map call
            dynamics_oracle = (
                lambda child_index: GeoGraphTimeline._calculate_node_dynamics(
                    child_index, node_map
                )
            )

            dynamics_type = self[time].df.index.map(dynamics_oracle)
            rel_growth, abs_growth = GeoGraphTimeline._calculate_growth_rates(node_map)

            self[time].df["relative_growth"] = rel_growth
            self[time].df["absolute_growth"] = abs_growth
            self[time].df["node_dynamic"] = dynamics_type

            # Detect dynamics type of unchanged/morphed by comparing to the
            # relative growth
            grew = (dynamics_type == "morphed/unchanged") & (rel_growth > 0)
            shrank = (dynamics_type == "morphed/unchanged") & (rel_growth < 0)
            unchanged = (dynamics_type == "morphed/unchanged") & (rel_growth == 0)
            # Update node dynamics for nodes in morphed/unchanged class
            self[time].df["node_dynamic"][grew] = "grew"
            self[time].df["node_dynamic"][shrank] = "shrank"
            self[time].df["node_dynamic"][unchanged] = "unchanged"

            return self[time].df["node_dynamic"]

    @staticmethod
    def _calculate_node_dynamics(child_index: int, nodemap: NodeMap) -> str:
        """
        Classify the node dynamics of the child node at `child_index` from the nodemap.

        The classified node dynamics is one of:
            `birth`: The child node has no ancestors in `nodemap`
            `split`: The child node was created as a split off of an ancestral node
            `unchanged/morphed`: The child node has exactly one ancestor
            `complex`: The child node has numerous ancestors and siblings
            `merged`: The child node has several ancestors and no siblings.

        Args:
            child_index (int): The index of the child node to calculate the dynamics
                for.

            nodemap (NodeMap): A node mapping between two GeoGraphs. Nodes in the source
                graph of the mapping are referred to as `ancestors`, nodes in the target
                graph are referred to as `children`.

        Returns:
            str: The identified node dynamics. One of `birth`, `split`,
                `morphed/unchanged`, `complex`, `merged`.
        """
        # TODO: numba typed dict for speedup
        ancestors = (~nodemap).mapping[child_index]

        if len(ancestors) == 0:
            # No ancestor means node birth in given timestep
            return "birth"
        elif len(ancestors) == 1:
            # check if split
            children = nodemap.mapping[ancestors[0]]
            if len(children) > 1:
                # One ancestor with several children means split
                return "split"
            else:
                # One ancestor with one child means morphed/unchanged
                return "morphed/unchanged"
        else:
            # Check if merged
            for ancestor in ancestors:
                children = nodemap.mapping[ancestor]
                if len(children) > 1:
                    # Several ancestors with several children means complex dynamics
                    return "complex"
            # Several ancestors with one children means merged
            return "merged"

    @staticmethod
    def _calculate_growth_rates(
        mapping: NodeMap,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return relative and absolute node growth rates for the nodes in `target_graph`
        of the given NodeMap

        Args:
            mapping (NodeMap): The maping of nodes between the the past (source) and
            future (target) graph

        Returns:
            Tuple[np.ndarray, np.ndarray]: The relative and absolute node growth rates
            in units of the CRS system that the graphs are in.
        """
        # TODO: numba for slight performance inmprovements

        backward_map = ~mapping
        relative_growth_rates = np.zeros(len(backward_map.mapping))
        absolute_growth_rates = np.zeros(len(backward_map.mapping))
        for i, (future, past) in enumerate(backward_map.mapping.items()):
            past_area = np.sum(backward_map.trg_graph.df.geometry.loc[past].area)
            future_area = np.sum(backward_map.src_graph.df.geometry.loc[future].area)

            relative_growth_rates[i] = (future_area - past_area) / (
                future_area + past_area
            )
            absolute_growth_rates[i] = future_area - past_area

        return relative_growth_rates, absolute_growth_rates
