"""Functions for calculating metrics from a GeoGraph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    import geograph


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


########################################################################################
###### 1. Landscape level metrics
########################################################################################
def _num_patches(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        value=len(geo_graph.df),
        name="num_patches",
        description="The number of patches in the graph.",
        variant="conventional",
        unit="dimensionless",
    )


def _avg_patch_area(geo_graph: geograph.GeoGraph) -> Metric:

    total_area = np.sum(geo_graph.df.area.values)
    num_patches = geo_graph.get_metric("num_patches").value

    return Metric(
        value=total_area / num_patches,
        name="avg_patch_area",
        description="The average area of the patches in the graph.",
        variant="conventional",
        unit="CRS.unit**2",
    )


def _total_area(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        value=np.sum(geo_graph.df.area.values),
        name="total_area",
        description="The total area of all the patches in the graph.",
        variant="conventional",
        unit="CRS.unit**2",
    )


def _patch_density(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        value=1.0 / geo_graph.get_metric("avg_patch_area").value,
        name="patch_density",
        description="Number of patches divided by total area of the graph.",
        variant="conventional",
        unit="1 / CRS.unit**2",
    )


def _largest_patch_index(geo_graph: geograph.GeoGraph) -> Metric:

    total_area = np.sum(geo_graph.df.area.values)
    max_patch_area = max(geo_graph.df.area.values)

    return Metric(
        value=max_patch_area / total_area,
        name="largest_patch_index",
        description="The proportion of landscape comprised by the largest patch.",
        variant="conventional",
        unit="dimensionless",
    )


def _shannon_diversity_index(geo_graph: geograph.GeoGraph) -> Metric:
    """
    Calculate shannon diversity of GeoGraph

    Further reference:
        https://pylandstats.readthedocs.io/en/latest/landscape.html
        https://en.wikipedia.org/wiki/Diversity_index
    """
    class_prop_of_landscape = np.array(
        [
            geo_graph.get_metric(
                "proportion_of_landscape", class_value=class_value
            ).value
            for class_value in geo_graph.classes
        ]
    )

    description = (
        "SHDI approaches 0 when the entire landscape consists of a single "
        "patch, and increases as the number of classes increases and/or the "
        "proportional distribution of area among classes becomes more equitable."
    )

    return Metric(
        value=-np.sum(class_prop_of_landscape * np.log(class_prop_of_landscape)),
        name="shannon_diversity_index",
        description=description,
        variant="conventional",
        unit="dimensionless",
    )


def _simpson_diversity_index(geo_graph: geograph.GeoGraph) -> Metric:
    """
    Calculate simpson diversity of GeoGraph

    Reference:
        umass.edu/landeco/teaching/landscape_ecology/schedule/chapter9_metrics.pdf
    """

    class_prop_of_landscape = np.array(
        [
            geo_graph.get_metric(
                "proportion_of_landscape", class_value=class_value
            ).value
            for class_value in geo_graph.classes
        ]
    )

    description = (
        "Probability that any two pixels drawn at random will be of different"
        "class types"
    )

    return Metric(
        value=1 - np.sum(class_prop_of_landscape ** 2),
        name="simpson_diversity_index",
        description=description,
        variant="conventional",
        unit="dimensionless",
    )


LANDSCAPE_METRICS_DICT = {
    "avg_patch_area": _avg_patch_area,
    "total_area": _total_area,
    "num_patches": _num_patches,
    "patch_density": _patch_density,
    "largest_patch_index": _largest_patch_index,
    "shannon_diversity_index": _shannon_diversity_index,
    "simpson_diversity_index": _simpson_diversity_index,
}

########################################################################################
###### 2. Landcover class level metrics
########################################################################################


def _class_total_area(geo_graph: geograph.GeoGraph, class_value: int) -> Metric:

    class_areas = geo_graph.df["geometry"][
        geo_graph.df["class_label"] == class_value
    ].area.values

    return Metric(
        value=np.sum(class_areas),
        name=f"total_area_class={class_value}",
        description=f"Total area of all patches of class {class_value} in the graph.",
        variant="conventional",
        unit="CRS.unit**2",
    )


def _class_avg_patch_area(geo_graph: geograph.GeoGraph, class_value: int) -> Metric:

    class_num_patches = geo_graph.get_metric(
        "num_patches", class_value=class_value
    ).value
    class_total_area = geo_graph.get_metric("total_area", class_value=class_value).value

    return Metric(
        value=class_total_area / class_num_patches,
        name=f"avg_patch_area_class={class_value}",
        description=f"The average area of patches of class {class_value} in the graph.",
        variant="conventional",
        unit="CRS.unit**2",
    )


def _class_num_patches(geo_graph: geograph.GeoGraph, class_value: int) -> Metric:
    return Metric(
        value=np.sum(geo_graph.df["class_label"] == class_value),
        name=f"num_patches_class={class_value}",
        description=f"The number of patches of class {class_value} in the graph.",
        variant="conventional",
        unit="dimensionless",
    )


def _class_proportion_of_landscape(
    geo_graph: geograph.GeoGraph, class_value: int
) -> Metric:

    class_total_area = geo_graph.get_metric("total_area", class_value=class_value).value
    total_area = geo_graph.get_metric("total_area").value

    return Metric(
        value=class_total_area / total_area,
        name=f"proportion_of_landscape_class={class_value}",
        description=f"The proportional abundance of {class_value} in the graph.",
        variant="conventional",
        unit="dimensionless",
    )


def _class_patch_density(geo_graph: geograph.GeoGraph, class_value: int) -> Metric:

    class_num_patches = geo_graph.get_metric(
        "num_patches", class_value=class_value
    ).value
    total_area = geo_graph.get_metric("total_area").value

    return Metric(
        value=class_num_patches / total_area,
        name=f"patch_density_class={class_value}",
        description=f"Density of patches of class {class_value} in the graph.",
        variant="conventional",
        unit="1 / CRS.unit**2",
    )


def _class_largest_patch_index(
    geo_graph: geograph.GeoGraph, class_value: int
) -> Metric:
    """
    Return proportion of total landscape comprised by largest patch of given class.

    Definition taken from:
        https://pylandstats.readthedocs.io/en/latest/landscape.html
    """

    total_area = geo_graph.get_metric("total_area").value
    class_areas = geo_graph.df["geometry"][
        geo_graph.df["class_label"] == class_value
    ].area

    description = (
        "Proportion of total landscape compriesed by largest patch of "
        f"class {class_value} in the graph.",
    )

    return Metric(
        value=max(class_areas) / total_area,
        name=f"patch_density_class={class_value}",
        description=description,
        variant="conventional",
        unit="dimensionless",
    )


def _class_total_edge(geo_graph: geograph.GeoGraph, class_value: int) -> Metric:
    # TODO: Implement option for not counting edges.

    class_perimeters = geo_graph.df["geometry"][
        geo_graph.df["class_label"] == class_value
    ].length

    return Metric(
        value=np.sum(class_perimeters),
        name=f"total_edge_class={class_value}",
        description=f"Total edgelength of patches of class {class_value} in the graph.",
        variant="conventional",
        unit="CRS.unit",
    )


def _class_edge_density(geo_graph: geograph.GeoGraph, class_value: int) -> Metric:
    """
    Return total length of class edges for the given class.
    Note: Currently the boundary also counted towards the edge length.

    Definition taken from:
        https://pylandstats.readthedocs.io/en/latest/landscape.html
    """
    # TODO: Implement option for not counting edges.

    total_edge = geo_graph.get_metric("total_edge", class_value=class_value).value
    total_area = geo_graph.get_metric("total_area", class_value=class_value).value

    description = (
        f"Edge length per unit area of patches of class {class_value} in the graph."
    )

    return Metric(
        value=total_edge / total_area,
        name=f"total_edge_density_class={class_value}",
        description=description,
        variant="conventional",
        unit="1 / CRS.unit",
    )


def _class_shape_index(geo_graph: geograph.GeoGraph, class_value: int) -> Metric:
    """
    Return shape index of given class.

    Definition taken from:
        https://pylandstats.readthedocs.io/en/latest/landscape.html
    """

    total_edge = geo_graph.get_metric("total_edge", class_value=class_value).value
    total_area = geo_graph.get_metric("total_area", class_value=class_value).value

    description = (
        "SI >=1 ; LSI equals 1 when the entire landscape consists of a single patch of "
        f"class {class_value}, and increases without limit as the patches of class "
        f"{class_value} become more disaggregated."
    )

    return Metric(
        value=0.25 * total_edge / np.sqrt(total_area),
        name=f"shape_index_class={class_value}",
        description=description,
        variant="conventional",
        unit="dimensionless",
    )


def _class_effective_mesh_size(
    geo_graph: geograph.GeoGraph, class_value: int
) -> Metric:
    """
    Return effective mesh size of given class.

    Definition taken from:
        https://pylandstats.readthedocs.io/en/latest/landscape.html
    """

    class_areas = geo_graph.df["geometry"][
        geo_graph.df["class_label"] == class_value
    ].area
    total_area = geo_graph.get_metric("total_area").value

    description = (
        "A <= MESH <= A ; MESH approaches its minimum when there is a single"
        " corresponding patch of one pixel, and approaches its maximum when the "
        "landscape consists of a single patch."
    )

    return Metric(
        value=np.sum(class_areas ** 2) / total_area,
        name=f"effective_mesh_size_class={class_value}",
        description=description,
        variant="conventional",
        unit="CRS.unit**2",
    )


CLASS_METRICS_DICT = {
    "num_patches": _class_num_patches,
    "avg_patch_area": _class_avg_patch_area,
    "total_area": _class_total_area,
    "proportion_of_landscape": _class_proportion_of_landscape,
    "patch_density": _class_patch_density,
    "largest_patch_index": _class_largest_patch_index,
    "total_edge": _class_total_edge,
    "edge_density": _class_edge_density,
    "shape_index": _class_shape_index,
    "effective_mesh_size": _class_effective_mesh_size,
}

########################################################################################
###### 3. Habitat componment level metrics
########################################################################################
def _num_components(geo_graph: geograph.GeoGraph) -> Metric:
    return Metric(
        value=nx.number_connected_components(geo_graph.graph),
        name="num_components",
        description="The number of connected components in the graph.",
        variant="component",
        unit="dimensionless",
    )


def _avg_component_area(geo_graph: geograph.GeoGraph) -> Metric:
    if not geo_graph.components.has_df:
        print("Calculating component polygons...")
        geo_graph.components = geo_graph.get_graph_components(calc_polygons=True)
    comp_geograph = geo_graph.components
    return Metric(
        value=np.mean(comp_geograph.df.area.values),
        name="avg_component_area",
        description="The average area of the components in the graph",
        variant="component",
        unit="CRS.unit**2",
    )


def _avg_component_isolation(geo_graph: geograph.GeoGraph) -> Metric:
    """
    Calculate the average distance to the next-nearest component.

    Warning: very computationally expensive for graphs with more than ~100
    components.
    """
    if not geo_graph.components.has_df or not geo_graph.components.has_distance_edges:
        print(
            """Warning: very computationally expensive for graphs with more
              than ~100 components."""
        )
        geo_graph.components = geo_graph.get_graph_components(
            calc_polygons=True, add_distance_edges=True
        )
    comp_geograph = geo_graph.components
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
        unit="CRS.unit",
    )


COMPONENT_METRICS_DICT = {
    "num_components": _num_components,
    "avg_component_area": _avg_component_area,
    "avg_component_isolation": _avg_component_isolation,
}


########################################################################################
###### 4. Define access interface for GeoGraph
########################################################################################

STANDARD_METRICS = ["num_components", "avg_patch_area", "total_area"]


def _get_metric(
    name: str,
    geo_graph: geograph.GeoGraph,
    class_value: Optional[int] = None,
    **metric_kwargs,
) -> Metric:
    """
    Calculate selected metric for the given GeoGraph

    Args:
        name (str): Name of the metric to compute
        geo_graph (geograph.GeoGraph): GeoGraph object to compute the metric for
        class_value (Optional[int], optional): The value of the desired class if a
            class level metric is desired. None if a landscape/component level metric
            is desired. Defaults to None.

    Returns:
        Metric: The desired metric
    """

    # Landscape level metrics
    if class_value is None:
        try:
            try:
                return LANDSCAPE_METRICS_DICT[name](geo_graph, **metric_kwargs)
            except KeyError as key_error:
                return COMPONENT_METRICS_DICT[name](geo_graph, **metric_kwargs)
        except KeyError as key_error:
            raise ValueError(
                f"Argument `{name}` is not a valid landscape/component metric.\n"
                "Available landscape metrics are: "
                f"\n{list(LANDSCAPE_METRICS_DICT.keys())}.\n"
                "Available component metrics are: "
                f"\n{list(COMPONENT_METRICS_DICT.keys())}."
            ) from key_error

    # Class level metrics
    else:
        try:
            return CLASS_METRICS_DICT[name](
                geo_graph, class_value=class_value, **metric_kwargs
            )
        except KeyError as key_error:
            raise ValueError(
                "Argument `name` is not a valid class metric. "
                f"Available class metrics are: \n{list(CLASS_METRICS_DICT.keys())}"
            ) from key_error
