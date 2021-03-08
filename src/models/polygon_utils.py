"""Helper functions for overlap computations with polygons in shapely."""
from typing import List

from geopandas.array import GeometryArray
from numpy import ndarray
from shapely.geometry.polygon import Polygon

# Note: All DE-9IM patterns below are streamlined to work well with polygons.
#  They are not guaranteed to work on lower dimensional objects (points/lines)
CORNER_ONLY_PATTERN = "FF*F0****"
EDGE_ONLY_PATTERN = "FF*F1****"
# Note: since we deal with polygons only, we can use a simplified overlap pattern:
#  If polygons overlap with more than just their edge, they will automatically overlap
#  with their interiors
OVERLAP_PATTERN = "T********"

# Create empty polygon
EMPTY_POLYGON = Polygon()


def de9im_match(pattern: str, target_pattern: str) -> bool:
    """
    Check a DE-9IM pattern `pattern` against a target DE-9IM pattern.

    Note:
        To enable maximal speed, patterns are not parsed for correctness. For
        correct patterns consult https://en.wikipedia.org/wiki/DE-9IM.

    Args:
        pattern (str): DE-9IM pattern to check as string
        target_pattern (str): DE-9IM pattern against which to check as string

    Returns:
        bool: True, iff pattern matches with target_pattern
    """
    for char, target_char in zip(pattern, target_pattern):
        if target_char == "*":
            continue
        elif target_char == "T" and char in "012":
            continue
        elif char == target_char:
            continue
        else:
            return False
    return True


def connect_with_interior_or_edge_or_corner(
    polygon1: Polygon, polygon2: Polygon
) -> bool:
    """
    Return True iff `polygon1` and `polygon2` overlap in interior, edges or corners.

    Args:
        polygon1 (Polygon): A shapely Polygon
        polygon2 (Polygon): The other shapely Polygon

    Returns:
        bool: True, iff `polygon1` and `polygon2` intersect.
    """
    return polygon1.intersects(polygon2)


def connect_with_interior_or_edge(polygon1: Polygon, polygon2: Polygon) -> bool:
    """
    Return True iff `polygon1` and `polygon2` overlap in interior/edge, but not corner.

    Args:
        polygon1 (Polygon): A shapely Polygon
        polygon2 (Polygon): The other shapely Polygon

    Returns:
        bool: True, iff `polygon1` and `polygon2` overlap in their interior/edge.
    """
    pattern = polygon1.relate(polygon2)
    return de9im_match(pattern, EDGE_ONLY_PATTERN) or de9im_match(
        pattern, OVERLAP_PATTERN
    )


def connect_with_interior(polygon1: Polygon, polygon2: Polygon) -> bool:
    """
    Return True iff `polygon1` and `polygon2` overlap in interior, but not edge/corner.

    Args:
        polygon1 (Polygon): A shapely Polygon
        polygon2 (Polygon): The other shapely Polygon

    Returns:
        bool: True, iff `polygon1` and `polygon2` overlap in their interior.
    """
    return polygon1.relate_pattern(polygon2, OVERLAP_PATTERN)


def connect_with_interior_or_edge_or_corner_bulk(
    polygon: Polygon, polygon_array: GeometryArray
) -> ndarray:
    """
    Return boolean array with True iff polygons overlap in interior, edges or corners.

    Args:
        polygon (Polygon): A shapely Polygon
        polygon_array (GeometryArray): The other shapely Polygons in a geopandas
            geometry array

    Returns:
        np.array: Boolean array with value True, iff `polygon` and the polygon in
            `polygon_array` at the given location intersect.
    """
    return polygon_array.intersects(polygon)


def connect_with_interior_or_edge_bulk(
    polygon: Polygon, polygon_array: GeometryArray
) -> List[bool]:
    """
    Return boolean array with True iff polys overlap in interior/edge, but not corner.

    Args:
        polygon (Polygon): A shapely Polygon
        polygon_array (GeometryArray): The other shapely Polygons in a geopandas
            geometry array

    Returns:
        List[bool]: Boolean array with value True, iff `polygon` and the polygon in
            `polygon_array` at the given location overlap in their interior/edge.
    """
    patterns = polygon_array.relate(polygon)
    return [
        de9im_match(pattern, EDGE_ONLY_PATTERN) or de9im_match(pattern, OVERLAP_PATTERN)
        for pattern in patterns
    ]


def connect_with_interior_bulk(
    polygon: Polygon, polygon_array: GeometryArray
) -> List[bool]:
    """
    Return boolean array with True iff polys overlap in interior, but not corner/edge.

    Args:
        polygon (Polygon): A shapely Polygon
        polygon_array (GeometryArray): The other shapely Polygons in a geopandas
            geometry array

    Returns:
        List[bool]: Boolean array with value True, iff `polygon` and the polygon in
            `polygon_array` at the given location overlap in their interior.
    """
    patterns = polygon_array.relate(polygon)
    return [de9im_match(pattern, OVERLAP_PATTERN) for pattern in patterns]


def collapse_empty_polygon(polygon: Polygon) -> Polygon:
    """
    Collapse `polygon` to an `EMPTY_POLYGON` if it is empty.

    Args:
        polygon (Polygon): The polygon to collapse if empty

    Returns:
        Polygon: Either the original, unchanges polygon or an empty polygon
    """
    if polygon.is_empty:
        return EMPTY_POLYGON
    else:
        return polygon
