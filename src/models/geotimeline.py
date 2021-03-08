"""Module for analysing multiple GeoGraph objects."""
from __future__ import annotations

from typing import Union, List, Dict
import datetime

from src.models.geograph import GeoGraph
from src.models.binary_graph_operations import identify_graphs, NodeMap

# type alias
TimeStamp = Union[int, datetime.datetime]


class NotCachedError(Exception):
    """Basic exception values which were not yet cached."""


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
    """Timeline of multiple GeoGraphs #TODO"""

    def __init__(self, data) -> None:

        # Initialize empty graphs dict
        self._graphs: Dict[TimeStamp, GeoGraph] = dict()

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
        return list(self._graphs.keys())

    def __getitem__(self, time: TimeStamp) -> GeoGraph:
        return self._graphs[time]

    def __len__(self) -> int:
        return len(self._graphs)

    def __iter__(self) -> GeoGraph:
        return iter(self._graphs.values())

    def _sort_by_time(self, reverse: bool = False) -> None:
        self._graphs = {
            time: self._graphs[time] for time in sorted(self._graphs, reverse=reverse)
        }

    def _load_from_sequence(self, graph_list: List[TimedGeoGraph]) -> None:

        # Make sure list is sorted in ascending time order (earliest = left)
        by_time = lambda item: item.time
        self._graphs = {graph.time: graph for graph in sorted(graph_list, key=by_time)}

    def _load_from_dict(self, graph_dict: Dict[TimeStamp, GeoGraph]):

        self._graphs = graph_dict
        self._sort_by_time()

    def identify(
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
        if (time1, time2) in self._node_map_cache.keys():
            return self._node_map_cache[(time1, time2)]
        elif (time2, time1) in self._node_map_cache.keys():
            map_from_inverse = self._node_map_cache[(time2, time1)].invert()
            self._node_map_cache[(time1, time2)] = map_from_inverse
            return map_from_inverse
        else:
            raise NotCachedError

    def empty_node_map_cache(self) -> None:
        self._node_map_cache = dict()

    def timestack(self, use_cached: bool = True) -> List[NodeMap]:
        node_maps = []
        for time1, time2 in zip(self.times, self.times[1:]):
            node_maps.append(self.identify(time1, time2, use_cached))
        return node_maps
