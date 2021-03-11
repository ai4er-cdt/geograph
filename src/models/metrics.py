"""Functions for calculating metrics from a GeoGraph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.models import geograph


# define a metric dataclass with < <= => > == comparisons that work as you would
# expect intuitively
@dataclass()
class Metric:
    """Class to represent a metric for a GeoGraph, with associated metadata."""

    value: Any  # No good Numpy type hints
    name: str
    description: str
    variant: Optional[str]
    unit: Optional[str] = None

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Metric):
            return False
        return self.value == o.value

    def __lt__(self, o: object) -> bool:
        if not isinstance(o, Metric):
            return False
        return self.value < o.value

    def __le__(self, o: object) -> bool:
        if not isinstance(o, Metric):
            return False
        return self.value <= o.value

    def __gt__(self, o: object) -> bool:
        if not isinstance(o, Metric):
            return False
        return self.value > o.value

    def __ge__(self, o: object) -> bool:
        if not isinstance(o, Metric):
            return False
        return self.value >= o.value


def _num_components(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        value=nx.number_connected_components(geo_graph.graph),
        name="num_components",
        description="The number of connected components in the graph.",
        variant="component",
    )


def _avg_patch_area(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        value=np.mean(geo_graph.df.area.values),
        name="avg_patch_area",
        description="The average area of the patches in the graph.",
        variant="conventional",
    )


def _total_area(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        value=np.sum(geo_graph.df.area.values),
        name="total_area",
        description="The total area of all the patches in the graph.",
        variant="conventional",
    )


def _avg_component_area(geo_graph: geograph.GeoGraph) -> Metric:
    comp_geograph = geo_graph.components
    if not comp_geograph.has_df:
        raise ValueError(
            "This metric is not valid for ComponentGeoGraphs without a dataframe."
        )
    return Metric(
        value=np.mean(comp_geograph.df.area.values),
        name="avg_component_area",
        description="The average area of the components in the graph",
        variant="component",
    )


def _avg_component_isolation(geo_graph: geograph.GeoGraph) -> Metric:
    """Calculate the average distance to the next-nearest component."""
    comp_geograph = geo_graph.components
    if not comp_geograph.has_df:
        raise ValueError(
            "This metric is not valid for ComponentGeoGraphs without a dataframe."
        )
    elif not comp_geograph.has_distance_edges:
        raise ValueError(
            "This metric is not valid for ComponentGeoGraphs without distance edges."
        )
    if len(comp_geograph.components_list) == 1:
        val: Any = 0
    else:
        dist_set = set()
        for comp in comp_geograph.graph.nodes:
            for nbr in comp_geograph.graph.adj[comp]:
                dist_set.update(
                    [
                        dist
                        for u, v, dist in comp_geograph.graph.edges(
                            nbr, data="distance"
                        )
                        if v != comp
                    ]
                )
        val = np.mean(np.fromiter(dist_set, np.float32, len(dist_set)))
    return Metric(
        value=val,
        name="avg_component_isolation",
        description="The average distance to the next-nearest component",
        variant="component",
    )


METRICS_DICT = {
    "num_components": _num_components,
    "avg_patch_area": _avg_patch_area,
    "total_area": _total_area,
    "avg_component_area": _avg_component_area,
    "avg_component_isolation": _avg_component_isolation,
}


def _get_metric(name: str, geo_graph: geograph.GeoGraph) -> Metric:
    try:
        return METRICS_DICT[name](geo_graph)
    except KeyError as key_error:
        raise ValueError("Argument `name` is not a valid metric") from key_error
