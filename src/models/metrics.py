"""Functions for calculating metrics from a GeoGraph."""
from dataclasses import dataclass
from typing import Any, List, Optional

import networkx as nx
import numpy as np
import shapely
from src.models import geograph


@dataclass
class Metric:
    name: str
    description: str
    value: Any  # No good Numpy type hints
    variant: Optional[str]
    unit: Optional[str] = None


def _get_num_components(geo_graph: geograph.GeoGraph) -> Metric:
    components: List[set] = list(nx.connected_components(geo_graph.graph))
    return Metric(
        name="num_components",
        description="The number of connected components in the graph.",
        value=len(components),
        variant="component",
    )


def _get_avg_patch_area(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        name="avg_patch_area",
        description="The average area of the patches in the graph.",
        value=np.mean(geo_graph.df.area.values),
        variant="conventional",
    )


def _get_total_area(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        name="total_area",
        description="The total area of all the patches in the graph.",
        value=np.sum(geo_graph.df.area.values),
        variant="conventional",
    )


def _get_avg_component_area(geo_graph: geograph.GeoGraph) -> Metric:
    components_gdf, _ = geo_graph.get_graph_components(geo_graph.graph)
    return Metric(
        name="avg_component_area",
        description="The average area of the components in the graph",
        value=np.mean(components_gdf.area.values),
        variant="component",
    )


def _get_avg_component_isolation(geo_graph: geograph.GeoGraph) -> Metric:
    """Return the average distance to the next-nearest component."""
    components_gdf, _ = geo_graph.get_graph_components(geo_graph.graph)
    comp_graph = nx.Graph()

    geom: List[shapely.Polygon] = components_gdf.geometry.tolist()
    graph_dict = {}

    # Creating nodes (=vertices) and finding neighbors
    for index, polygon in enumerate(geom):
        neighbours = components_gdf.sindex.query(polygon, predicate="intersects")

        graph_dict[index] = neighbours
        # add each component as a node to the graph with useful attributes
        comp_graph.add_node(
            index,
            rep_point=polygon.representative_point(),
            area=polygon.area,
            perimeter=polygon.length,
            bounds=polygon.bounds,
            geometry=polygon,
        )
    # iterate through the dict and add edges between neighbouring components
    for polygon_id, neighbours in graph_dict.items():
        for neighbour_id in neighbours:
            if polygon_id != neighbour_id:
                comp_graph.add_edge(polygon_id, neighbour_id)

    dist_set = set()
    for comp in comp_graph.nodes:
        for neighbour in comp_graph.adj[comp]:
            polygon = comp.geometry
            dist_set.update(
                [polygon.distance(nbr.geometry) for nbr in comp_graph.adj[neighbour]]
            )
    mean_patch_isolation = np.mean(np.fromiter(dist_set, np.float32, len(dist_set)))
    return Metric(
        name="avg_component_isolation",
        description="The average distance to the next-nearest component",
        value=mean_patch_isolation,
        variant="component",
    )


METRICS_DICT = {
    "num_components": _get_num_components,
    "avg_patch_area": _get_avg_patch_area,
    "total_area": _get_total_area,
    "avg_component_area": _get_avg_component_area,
    "avg_component_isolation": _get_avg_component_isolation,
}


def get_metric(name: str, geo_graph: geograph.GeoGraph) -> Metric:
    try:
        return METRICS_DICT[name](geo_graph)
    except KeyError as key_error:
        raise ValueError("Argument `name` is not a valid metric") from key_error
