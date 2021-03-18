"""A collection of utility functions for plotting landcover datasets"""
import numpy as np
import pandas as pd
from numba import njit
from src.constants import ESA_LANDCOVER_DIR

# from src.data_loading.landcover_plot_utils import classes_to_rgb, ESA_SUPER_RGB_ARRAY


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


#           new_class: old classes                      : R, G, B

_eunis_map = """0 : 0                                    : 0, 0, 0
            1 : 21, 22, 23, 24, 29, 30, 31, 33          : 255, 255, 100
            2 : 2                                       : 0, 160, 0
            3 : 3                                       : 0, 100, 0
            4 : 4                                       : 0, 120, 90
            5 : 38, 43                                  : 190, 150, 0
            6 : 35, 36, 37                              : 150, 100, 0
            7 : 25, 26, 27, 28                          : 195, 20, 0
            8 : 1                                       : 255, 220, 210
            9 : 5                                       : 0, 70, 200
            10 : 6                                      : 0, 220, 130
            11 : 7, 8                                   : 0, 0, 0
            12: 9, 11                                   : 0, 0, 0
            13 : 16, 17, 18, 32                         : 0, 0, 0
            14 : 19, 34                                 : 0, 0, 0
            15: 39, 40                                  : 0, 0, 0
            16: 41                                      : 0, 0, 0
            17: 42                                      : 0, 0, 0
            18: 12, 13, 14, 15, 20                      : 0, 0, 0
            19: 10                                      : 0, 0, 0"""
EUNIS_SUPER_LOL = [
    [[int(z) for z in y.split(",")] for y in x.split(":")]
    for x in _eunis_map.split("\n")
]


_esa_super_map = """0 : 0                                : 0, 0, 0
            1 : 10, 11, 12, 20, 30, 40                  : 255, 255, 100
            2 : 60, 61, 62, 80, 81, 82, 90, 100         : 0, 160, 0
            3 : 50, 70, 71, 72                          : 0, 100, 0
            4 : 160, 170                                : 0, 120, 90
            5 : 110, 130                                : 190, 150, 0
            6 : 120, 121, 122                           : 150, 100, 0
            7 : 190                                     : 195, 20, 0
            8 : 140, 150, 152, 153, 200, 201, 202, 220  : 255, 220, 210
            9 : 210                                     : 0, 70, 200
            10: 180                                     : 0, 220, 130"""
ESA_SUPER_LOL = [
    [[int(z) for z in y.split(",")] for y in x.split(":")]
    for x in _esa_super_map.split("\n")
]


def _sup_class_df_from_lol(lol: list = ESA_SUPER_LOL) -> pd.DataFrame:
    reduced_list = pd.DataFrame(lol).drop(columns=[1]).values.tolist()
    tmp_list = [[x[0][0], x[1][0], x[1][1], x[1][2]] for x in reduced_list]
    tmp_list.insert(0, ["NB_LAB", "R", "G", "B"])
    df = pd.DataFrame(tmp_list)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.set_index("NB_LAB")
    return df


ESA_SUPER_CLASSES = _sup_class_df_from_lol(ESA_SUPER_LOL)
ESA_SUPER_RGB_ARRAY = _class_rgb_array_from_df(ESA_SUPER_CLASSES)
EUNIS_SUPER_CLASSES = _sup_class_df_from_lol(EUNIS_SUPER_LOL)
EUNIS_SUPER_RGB_ARRAY = _class_rgb_array_from_df(EUNIS_SUPER_CLASSES)


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
