"""Contains tools for binary operations between GeoGraph objects"""
from typing import List
from numpy import ndarray
from src.models.polygon_utils import (
    connect_with_interior,
    connect_with_interior_or_edge,
    connect_with_interior_or_edge_or_corner,
    connect_with_interior_bulk,
    connect_with_interior_or_edge_bulk,
    connect_with_interior_or_edge_or_corner_bulk,
)

# For switching identifiction mode in `identify_node`
_SPATIAL_IDENTIFICATION_FUNCTION = {
    "corner": connect_with_interior_or_edge_or_corner,
    "edge": connect_with_interior_or_edge,
    "interior": connect_with_interior,
}
_BULK_SPATIAL_IDENTIFICATION_FUNCTION = {
    "corner": connect_with_interior_or_edge_or_corner_bulk,
    "edge": connect_with_interior_or_edge_bulk,
    "interior": connect_with_interior_bulk,
}


def identify_node(node: dict, other_graph: "GeoGraph", mode: str = "corner") -> ndarray:
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
        np.ndarray: List of node ids in `other_graph` which identify with `node`.
    """

    # Mode switch
    assert mode in ["corner", "edge", "interior"]
    have_valid_overlap = _BULK_SPATIAL_IDENTIFICATION_FUNCTION[mode]

    # Extract relevant node elements for legibility
    poly = node["geometry"]
    label = node["class_label"]

    # Get potential candidates for overlap
    candidate_ids = other_graph.rtree.query(poly)
    # Filter candidates according to the same class label
    # fmt: off
    candidate_ids = candidate_ids[
        other_graph._class_label(candidate_ids) == label  # pylint: disable=protected-access
    ]
    # Filter candidates accroding to correct spatial overlap
    # fmt: on
    candidate_ids = candidate_ids[
        have_valid_overlap(
            poly,
            other_graph._geometry(candidate_ids),  # pylint: disable=protected-access
        )
    ]

    return candidate_ids


### Deprecated but kept for tests and backward compatibility
def identify_node_old(
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

        candidate_node = other_graph.df.iloc[candidate_id]
        have_same_class_label = node["class_label"] == candidate_node["class_label"]

        if have_same_class_label and have_valid_overlap(
            node["geometry"], candidate_node["geometry"]
        ):
            identifies_with.append(candidate_id)

    return identifies_with
