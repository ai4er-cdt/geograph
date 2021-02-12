"""
This module contains functions that supply simple test data.
"""

from typing import Tuple

import geopandas as gpd
import shapely

from src.constants import CHERNOBYL_COORDS_UTM35N, UTM35N


def get_square_row(
    num_squares: int = 2,
    translation: Tuple[float, float] = (2, 0),
    square_side_len: float = 1,
    origin: Tuple[float, float] = (0, 0),
    crs: str = UTM35N,
) -> gpd.GeoDataFrame:
    """
    Creates a row of squares with side length `square_side_len`,
    each square is translated to the previous by `translation`.
    Starts with the bottom left corner of first square at `origin`.

    Args:
        num_squares (int, optional): number of squares. Defaults to 2.
        translation (Tuple[float, float], optional): vector that translates one square
            to the previous. Defaults to (2, 0).
        square_side_len (float, optional): length of the side of a square.
            Defaults to 1.
        origin (Tuple[float, float], optional): coordinates of bottom left corner of
            first square in row. Defaults to (0, 0).
        crs (str, optional): coordinate reference system. Should match the one used to
            to set origin. Defaults to UTM35N.

    Returns:
        gpd.GeoDataFrame: dataframe that contains the row of squares as polygons
    """

    out_gdf = gpd.GeoDataFrame(columns=["id", "geometry", "area"])

    # setting up the coordinates of first square
    square_coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    # scaling up by square_side_len
    square_coords = [
        tuple([coord * square_side_len for coord in coords]) for coords in square_coords
    ]
    # setting to origin coords
    square_coords = [[sum(x) for x in zip(coords, origin)] for coords in square_coords]

    square = shapely.geometry.Polygon(square_coords)

    # creating the row of squares
    for i in range(num_squares):
        xoff = i * translation[0]
        yoff = i * translation[1]
        tmp_square = shapely.affinity.translate(square, xoff=xoff, yoff=yoff)
        out_gdf.loc[i] = [i, tmp_square, tmp_square.area]

    out_gdf = out_gdf.set_crs(crs)

    return out_gdf


def get_polygon_gdf(name: str = "squares_apart") -> gpd.GeoDataFrame:
    """Load different test cases consisting of rows of squares.

    Args:
        name (str, optional): test data name. Defaults to "squares_apart".
            The options are:
            - "squares_overlapping": overlapping square pair at origin (0,0)
            - "squares_touching": touching square pair at origin (0,0)
            - "squares_apart": separated square pair at origin (0,0)
            - "chernobyl_squares_apart" and "chernobyl_squares_touching": similar to
                above but with origin at Chernobyl power reactor.

    Raises:
        ValueError: test data name is not in options above

    Returns:
        gpd.GeoDataFrame: dataframe with polygon test data
    """

    if name == "squares_overlapping":
        data = get_square_row(
            num_squares=2, translation=(0.5, 0), square_side_len=1, origin=(0, 0)
        )
    elif name == "squares_touching":
        data = get_square_row(
            num_squares=2, translation=(1, 0), square_side_len=1, origin=(0, 0)
        )
    elif name == "chernobyl_squares_apart":
        data = get_square_row(
            num_squares=2,
            translation=(200000, 0),
            square_side_len=100000,
            origin=CHERNOBYL_COORDS_UTM35N,
            crs=UTM35N,
        )
    elif name == "chernobyl_squares_touching":
        data = get_square_row(
            num_squares=2,
            translation=(100000, 0),
            square_side_len=100000,
            origin=CHERNOBYL_COORDS_UTM35N,
            crs=UTM35N,
        )
    elif name == "squares_apart":
        data = get_square_row(
            num_squares=2, translation=(2, 0), square_side_len=1, origin=(0, 0)
        )
    else:
        raise ValueError("The test data name does not match any existing data.")

    return data
