"""A collection of utility functions for plotting landcover datasets"""
import numpy as np
import pandas as pd
from numba import njit
from src.constants import ESA_LANDCOVER_DIR


def _class_rgb_array_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Convert class legend to array which holds RGB values for class i at i-th index.

    Note: This is an auxillary function, specificially written for the ESA CCI landcover
     classes. To use this function, adhere to the ESA CCI Legend conventions.

    Args:
        df (pd.DataFrame): dataframe holding the class information and rgb information
            for each class. Must be indexed by the class index. Must contain `R`, `G`,
            `B` columns for the RGB values.

    Returns:
        np.ndarray: A numpy array of RGB values (stored as np.uint8). The value at
            index i corresponds to the RGB value of class i.
    """

    # Initialize empty array with as many entries as the highest class index.
    # Array is will hold rgb value of class i at index i (needed for efficient
    #  looping with numba)
    class_rgb_array = np.empty((df.index.max() + 1, 3), dtype=np.uint8)

    for idx, row in df.iterrows():
        class_rgb_array[idx, ...] = np.array((row.R, row.G, row.B), dtype=np.uint8)

    return class_rgb_array


# Load the ESA CCI landcover classes and their RGB values as defaults
ESA_CCI_LEGEND = ESA_LANDCOVER_DIR / "ESACCI-LC-Legend.csv"
ESA_CCI_CLASSES = pd.read_csv(ESA_CCI_LEGEND, delimiter=";", index_col=0)
ESA_CCI_RGB_ARRAY = _class_rgb_array_from_df(ESA_CCI_CLASSES)


@njit()
def classes_to_rgb(
    data: np.ndarray, class_to_rgb: np.ndarray = ESA_CCI_RGB_ARRAY
) -> np.ndarray:
    """
    Convert array containing class indices into array containing RGB values.

    This function is meant to turn an array of landcover class indices to an array
    containing RGB values with a given `class_to_rgb` mapping that contains the RGB
    colors for class i at index i. Meant for plotting purposes with matplotlib's
    imshow.

    Args:
        data (np.ndarray): array with values corresponding to class indices. Values
            must be integers.
        class_to_rgb (np.ndarray, optional): array where the value of the i-th index
            corresponds to the RGB colors of class i. Defaults to ESA_CCI_RGB_ARRAY.

    Returns:
        np.ndarray: array of the same X-Y shape as original data, but with a third axis
            containing the RGB colors. Can be used for plotting directly with
            matplotlib's imshow.
    """

    n_rows, n_cols = data.shape
    rgb_data = np.empty((n_rows, n_cols, 3), dtype=np.uint8)

    for i in range(n_rows):
        for j in range(n_cols):
            rgb_data[i, j, ...] = class_to_rgb[data[i, j]]

    return rgb_data
