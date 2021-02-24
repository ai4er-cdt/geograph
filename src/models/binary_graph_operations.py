"""Contains tools for binary operations between GeoGraph objects"""
from typing import List
from src.models.polygon_utils import (
    connect_with_interior,
    connect_with_interior_or_edge,
    connect_with_interior_or_edge_or_corner,
)

# For switching identifiction mode in `identify_node`
_SPATIAL_IDENTIFICATION_FUNCTION = {
    "corner": connect_with_interior_or_edge_or_corner,
    "edge": connect_with_interior_or_edge,
    "interior": connect_with_interior,
}


def identify_node(
    node: dict, other_graph: "GeoGraph", mode: str = "corner"
) -> List[int]:
    """
    Return list of all node ids in `other_graph` which identify with the given `node`.

    Args:
        node (dict): The node for which to find nodes in `other_graphs` that can be
            identified with `node`.
        other_graph (GeoGraph): The GeoGraph object in which to search for
            identifications
        mode (str, optional): Must be one of `corner`, `edge` or `interior`. Defaults
            to "corner".
            The different modes correspond to different rules for identification:
            - corner: Polygons of the same `class_label` which overlap, touch in their
                edges or corners will be identified with each other. (fastest)
            - edge: Polygons of the same `class_label` which overlap or touch in their
                edges will be identified with each other.
            - interior: Polygons of the same `class_label` which overlap will be
                identified with each other. Touching corners or edges are not counted.

    Returns:
        List[int]: List of node ids in `other_graph` which identify with `node`.
    """

    # Mode switch
    assert mode in ["corner", "edge", "interior"]
    have_valid_overlap = _SPATIAL_IDENTIFICATION_FUNCTION[mode]

    # Build list of nodes in `other_graph` which identify with `node`
    identifies_with = []
    for candidate_id in other_graph.rtree.intersection(node["geometry"].bounds):

        candidate_node = other_graph.graph.nodes[candidate_id]
        have_same_class_label = node["class_label"] == candidate_node["class_label"]

        if have_same_class_label and have_valid_overlap(
            node["geometry"], candidate_node["geometry"]
        ):
            identifies_with.append(candidate_id)

    return identifies_with
