"""Helper functions for overlap computations with polygons in shapely"""
from numpy import ndarray
from shapely.geometry.polygon import Polygon
from geopandas.geoseries import GeometryArray

# Note: All DE-9IM patterns below are streamlined to work well with polygons.
#  They are not guaranteed to work on lower dimensional objects (points/lines)
CORNER_ONLY_PATTERN = "FF*F0****"
EDGE_ONLY_PATTERN = "FF*F1****"
# Note: since we deal with polygons only, we can use a simplified overlap pattern:
#  If polygons overlap with more than just their edge, they will automatically overlap
#  with their interiors
OVERLAP_PATTERN = "T********"


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


def connect_with_interior_or_edge_or_corner(poly1: Polygon, poly2: Polygon) -> bool:
    """
    Return True iff `poly1` and `poly2` overlap in their interior, edges or corners.

    Args:
        poly1 (Polygon): A shapely Polygon
        poly2 (Polygon): The other shapely Polygon

    Returns:
        bool: True, iff `poly1` and `poly2` intersect.
    """
    return poly1.intersects(poly2)


def connect_with_interior_or_edge(poly1: Polygon, poly2: Polygon) -> bool:
    """
    Return True iff `poly1` and `poly2` overlap in their interior/edge, but not corner.

    Args:
        poly1 (Polygon): A shapely Polygon
        poly2 (Polygon): The other shapely Polygon

    Returns:
        bool: True, iff `poly1` and `poly2` overlap in their interior/edge.
    """
    pattern = poly1.relate(poly2)
    return de9im_match(pattern, EDGE_ONLY_PATTERN) or de9im_match(
        pattern, OVERLAP_PATTERN
    )


def connect_with_interior(poly1: Polygon, poly2: Polygon) -> bool:
    """
    Return True iff `poly1` and `poly2` overlap in their interior, but not edge/corner.

    Args:
        poly1 (Polygon): A shapely Polygon
        poly2 (Polygon): The other shapely Polygon

    Returns:
        bool: True, iff `poly1` and `poly2` overlap in their interior.
    """
    return poly1.relate_pattern(poly2, OVERLAP_PATTERN)


def connect_with_interior_or_edge_or_corner_bulk(
    poly: Polygon, poly_array: GeometryArray
) -> ndarray:
    """
    Return boolean array with True iff polygons overlap in interior, edges or corners.

    Args:
        poly (Polygon): A shapely Polygon
        poly_array (GeometryArray): The other shapely Polygons in a geopandas geometry
            array

    Returns:
        np.array: Boolean array with value True, iff `poly` and the polygon in
            `poly_array` at the given location intersect.
    """
    return poly_array.intersects(poly)


def connect_with_interior_or_edge_bulk(
    poly: Polygon, poly_array: GeometryArray
) -> ndarray:
    """
    Return boolean array with True iff polys overlap in interior/edge, but not corner.

    Args:
        poly (Polygon): A shapely Polygon
        poly_array (GeometryArray): The other shapely Polygons in a geopandas geometry
            array

    Returns:
        np.array: Boolean array with value True, iff `poly` and the polygon in
            `poly_array` at the given location overlap in their interior/edge.
    """
    patterns = poly_array.relate(poly)
    return [
        de9im_match(pattern, EDGE_ONLY_PATTERN) or de9im_match(pattern, OVERLAP_PATTERN)
        for pattern in patterns
    ]


def connect_with_interior_bulk(poly: Polygon, poly_array: GeometryArray) -> ndarray:
    """
    Return boolean array with True iff polys overlap in interior, but not corner/edge.

    Args:
        poly (Polygon): A shapely Polygon
        poly_array (GeometryArray): The other shapely Polygons in a geopandas geometry
            array

    Returns:
        np.array: Boolean array with value True, iff `poly` and the polygon in
            `poly_array` at the given location overlap in their interior.
    """
    patterns = poly_array.relate(poly)
    return [de9im_match(pattern, OVERLAP_PATTERN) for pattern in patterns]
