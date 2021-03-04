"""Module for analysing multiple GeoGraph objects."""
from __future__ import annotations

from typing import Union, List, Dict
import datetime

from src.models.geograph import GeoGraph

# type alias
TimeStamp = Union[int, datetime.datetime]


class NodeMapping:
    """Class to store node mappings between two graph (the from_graph and to_graph)"""

    def __init__(
        self, from_graph: GeoGraph, to_graph: GeoGraph, mapping=Dict[int, List[int]]
    ):
        self._from_graph = from_graph
        self._to_graph = to_graph
        self._mapping = mapping

    @property
    def from_graph(self):
        return self._from_graph

    @property
    def to_graph(self):
        return self._to_graph

    @property
    def mapping(self):
        return self._mapping

    def __invert__(self):
        return self.invert()

    def __eq__(self, other: NodeMapping) -> bool:
        return (
            self.from_graph == other.from_graph
            and self.to_graph == other.to_graph
            and self.mapping == other.mapping
        )

    def invert(self) -> NodeMapping:
        inverted_mapping = {index: [] for index in self.to_graph.df.index}

        for from_node in self.from_graph.df.index:
            for to_node in self.mapping[from_node]:
                inverted_mapping[to_node].append(from_node)

        return NodeMapping(self.to_graph, self.from_graph, inverted_mapping)


class TimedGeoGraph(GeoGraph):
    def __init__(self, time: TimeStamp, **geographargs) -> None:
        super().__init__(**geographargs)
        self._time = time

    @property
    def time(self):
        return self._time


class GeoGraphTimeline:
    """Timelines of multiple GeoGraphs #TODO"""

    def __init__(self, data) -> None:

        if isinstance(data, list):
            self._load_from_sequence(graph_list=data)
        elif isinstance(data, dict):
            self._load_from_dict(graph_dict=data)
        else:
            raise NotImplementedError

    @property
    def times(self):
        return list(self._graphs.keys())

    def __getitem__(self, time):
        return self._graphs[time]

    def __len__(self):
        return len(self._graphs)

    def __iter__(self):
        return iter(self._graphs.values())

    def _sort_by_time(self, reverse: bool = False):
        self._graphs = {
            time: self._graphs[time] for time in sorted(self._graphs, reverse=reverse)
        }

    def _load_from_sequence(self, graph_list: List[TimedGeoGraph]):

        # Make sure list is sorted in ascending time order (earliest = left)
        by_time = lambda item: item.time
        self._graphs = {graph.time: graph for graph in sorted(graph_list, key=by_time)}

    def _load_from_dict(self, graph_dict: Dict[TimeStamp, GeoGraph]):

        self._graphs = graph_dict
        self._sort_by_time()

    def identify_nodes(self):
        raise NotImplementedError
