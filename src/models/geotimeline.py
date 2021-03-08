"""Module for analysing multiple GeoGraph objects."""
from __future__ import annotations

from typing import Union, List, Dict
import datetime

from src.models.geograph import GeoGraph

# type alias
TimeStamp = Union[int, datetime.datetime]


class NodeMap:
    """Class to store node mappings between two graphs (the src_graph and trg_graph)"""

    def __init__(
        self, src_graph: GeoGraph, trg_graph: GeoGraph, mapping: Dict[int, List[int]]
    ):
        """
        Class to store node mappings between two graphs (`trg_graph` and `src_graph`)

        This class stores a dictionary of node one-to-many relationships of nodes from
        `src_graph` to `trg_graph`. It also provides support for convenient methods for
        inverting the mapping and bundles the mapping information with references to
        the `src_graph` and `trg_graph`

        Args:
            src_graph (GeoGraph): Domain of the node map (keys in `mapping` correspond
                to indices from the `src_graph`).
            trg_graph (GeoGraph): Image of the node map (values in `mapping` correspond
                to indices from the `trg_graph`)
            mapping (Dict[int, List[int]], optional): A lookup table for the map which
                maps nodes form `src_graph` to `trg_graph`.
        """
        self._src_graph = src_graph
        self._trg_graph = trg_graph
        self._mapping = mapping

    @property
    def src_graph(self) -> GeoGraph:
        """Keys in the mapping dict correspond to node indices in the `src_graph`"""
        return self._src_graph

    @property
    def trg_graph(self) -> GeoGraph:
        """Values in the mapping dict correspond to node indices in the `trg_graph`"""
        return self._trg_graph

    @property
    def mapping(self) -> Dict[int, List[int]]:
        """
        Look-up table connecting node indices from `src_graph` to those of `trg_graph`.
        """
        return self._mapping

    def __invert__(self) -> NodeMap:
        """Compute the inverse NodeMap"""
        return self.invert()

    def __eq__(self, other: NodeMap) -> bool:
        """Check two NodeMaps for equality"""
        return (
            self.src_graph == other.src_graph
            and self.trg_graph == other.trg_graph
            and self.mapping == other.mapping
        )

    def invert(self) -> NodeMap:
        """Compute the inverse NodeMap from `trg_graph` to `src_graph`"""
        inverted_mapping = {index: [] for index in self.trg_graph.df.index}

        for src_node in self.src_graph.df.index:
            for trg_node in self.mapping[src_node]:
                inverted_mapping[trg_node].append(src_node)

        return NodeMap(
            src_graph=self.trg_graph, trg_graph=self.src_graph, mapping=inverted_mapping
        )


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
