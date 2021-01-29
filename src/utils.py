"""General project util functions"""
from sys import getsizeof
import requests


def human_readable_size(num: int, suffix: str = "B") -> str:
    """
    Convert a number of bytes into human readable format.

    This function is useful in conjuction with sys.getsizeof.

    Args:
        num (int): The number of bytes to convert
        suffix (str, optional): The suffix to use for bytes. Defaults to 'B'.

    Returns:
        str: A human readable version of the number of bytes.
    """
    assert num >= 0, "Size cannot be negative."
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if num < 1024:
            return f"{num:.0f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:1f} Y{suffix}"


def get_byte_size(obj: object) -> str:
    """
    Return human readable size of a python object in bytes.

    Args:
        obj (object): The object to analyse

    Returns:
        str: Human readable string with the sieze of the object
    """

    return human_readable_size(getsizeof(obj))


def get_osm_polygon(polygon_id: int, out_format: str = "geojson") -> str:
    """
    Return URL Query string for Open Street Map Polygon query

    Queries via api: http://polygons.openstreetmap.fr/

    Args:
        polygon_id (int): ID of object to be queried
        out_format (str, optional): Data format to request. Must be one of
            ["geojson", "wkt", "poly"]. Defaults to "geojson".

    Returns:
        str: The completed url query string for use via geopandas or requests
    """
    # Polygons are only generated upon first request by polygons.openstreaetmap.
    # This request makes sure the polygon of the requested ID was created before.
    requests.get(f"http://polygons.openstreetmap.fr/?id={polygon_id}")

    allowed = ["geojson", "wkt", "poly"]
    assert out_format in allowed, f"Format {out_format} must be one of {allowed}"

    return (
        f"http://polygons.openstreetmap.fr/get_{out_format}.py?id={polygon_id}&params=0"
    )
