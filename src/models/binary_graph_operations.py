"""Contains tools for binary operations between GeoGraph objects."""
from numpy import ndarray
from src.models.polygon_utils import (
    connect_with_interior_bulk, connect_with_interior_or_edge_bulk,
    connect_with_interior_or_edge_or_corner_bulk)

# For switching identifiction mode in `identify_node`
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

    # Get potential candidates for overlap
    candidate_ids = other_graph.rtree.query(node["geometry"], sort=True)
    # Filter candidates according to the same class label
    candidate_ids = candidate_ids[
        other_graph.class_label[candidate_ids] == node["class_label"]
    ]
    # Filter candidates accroding to correct spatial overlap
    candidate_ids = candidate_ids[
        have_valid_overlap(node["geometry"], other_graph.geometry[candidate_ids])
    ]

    return candidate_ids
