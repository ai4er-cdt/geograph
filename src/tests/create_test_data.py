"""Script to quickly and reproducibly create test data

Note: We may want to delete this at some point.
"""
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import numpy as np

from src.constants import SRC_PATH
from src.utils.rasterio_utils import polygonise
from src.tests.utils import get_array_transform, polygonise_sub_array


def _polygonise_splits(
    arr: np.ndarray, named_slices: Iterable[Dict[str, Tuple]]
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Create polygons from multiple sub-arrays of the given array.

    Note:
        Indices for `named_slices` must be given in x-y convention.
        x-y indexing is used for convenience with plotting later. The origin for
        x-y indexing is taken to be at the lower left corner of the array. The
        x-index increases horizontally to the right, y-index increases vertically to
        the top.

    Args:
        arr (np.ndarray): The array from which to select sub-arrays and polygonise them
        named_slices (Iterable[Dict[str, Tuple]]): An iterable of dictionaries
            containing the x-y limits of the sub-arrays to polygonise. x-y indices
            must be >= 0 or None.

    Returns:
        Dict[str, gpd.GeoDataFrame]: [description]
    """

    result = {}
    for name, (x_lims, y_lims) in named_slices.items():
        result[name] = polygonise_sub_array(arr, x_lims, y_lims)

    return result


if __name__ == "__main__":
    print("Creating test data ... ")
    TEST_DATA_FOLDER = SRC_PATH / "tests" / "testdata"
    TEST_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # 1. Create non-overlapping polygons
    print("(1/3) Create non-overlapping polygon data")
    # Create array
    np.random.seed(28)
    arr1 = np.random.randint(low=1, high=5, size=(6, 4), dtype=np.uint8)
    # Define splits
    splits_of_interest = {
        "lower_left": ((0, 2), (0, 3)),
        "upper_right": ((2, 4), (3, 6)),
        "upper_left": ((0, 2), (3, 6)),
        "lower_right": ((2, 4), (0, 3)),
    }
    # Poligonise
    polygons1 = _polygonise_splits(arr1, splits_of_interest)
    polygons1["full"] = polygonise(arr1, transform=get_array_transform(arr1))
    # Save
    for save_name, df in polygons1.items():
        save_path = TEST_DATA_FOLDER / "adjacent" / f"{save_name}.gpkg"
        df.to_file(save_path, driver="GPKG")

    # 2. Create overlapping polygons
    print("(2/3) Create overlapping polygon data")
    # Create array
    np.random.seed(285)
    arr2 = np.random.randint(low=1, high=5, size=(4, 4), dtype=np.uint8)
    # Define splits
    splits_of_interest = {
        "lower_left": ((0, 3), (0, 3)),
        "upper_right": ((1, 4), (1, 4)),
        "upper_left": ((0, 3), (1, 4)),
        "lower_right": ((1, 4), (0, 3)),
    }
    # Polygonise
    polygons2 = _polygonise_splits(arr2, splits_of_interest)
    polygons2["full"] = polygonise(arr2, transform=get_array_transform(arr2))
    # Save
    for save_name, df in polygons2.items():
        save_path = TEST_DATA_FOLDER / "overlapping" / f"{save_name}.gpkg"
        df.to_file(save_path, driver="GPKG")

    # 3. Create time-stacked polygons
    print("(3/3) Create time slice data")
    # Settings
    np.random.seed(184)
    # Create polygons
    for i in range(5):
        arr_t = np.random.randint(low=1, high=4, size=(4, 4), dtype=np.uint8)
        polygons_t = polygonise(arr_t, transform=get_array_transform(arr_t))
        save_path = TEST_DATA_FOLDER / "timestack" / f"time_{i}.gpkg"
        polygons_t.to_file(save_path, driver="GPKG")
