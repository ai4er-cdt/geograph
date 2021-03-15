"""
esa-compress.py
================================================================
# from src.preprocessing.esa_comp import compress_esa, decompress_esa, FORW_D, REV_D

from src.preprocessing.esa_compress import esa_to_superclasses, eunis_to_superclasses

"""
from typing import Tuple
import numpy as np
import xarray as xr


def _make_esa_map_d() -> Tuple[dict, dict]:
    """
    This function creats the mapping between the esa cci habitat labels and
    a reduced set of the same length. Only usable if the same habitat labels as in the
    Chernobyl region are used.
    :return: forw_d, rev_d; two dictionaries for the forwards / reverse transformation.
    """
    a = [
        0,
        10,
        11,
        30,
        40,
        60,
        61,
        70,
        80,
        90,
        100,
        110,
        130,
        150,
        160,
        180,
        190,
        200,
        201,
        210,
    ]
    b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    forw_d = {}
    rev_d = {}
    for i in range(len(a)):
        forw_d[a[i]] = b[i]
        rev_d[b[i]] = a[i]
    return forw_d, rev_d


FORW_D, REV_D = _make_esa_map_d()  # Makes global objects for the mapping dicts.


# Create the unvectorised functions.


def _compress_esa(x: int) -> int:
    return FORW_D[x]


def _decompress_esa(x: int) -> int:
    return REV_D[x]


# Vectorize the functions

compress_esa = np.vectorize(_compress_esa)
decompress_esa = np.vectorize(_decompress_esa)


def esa_to_superclasses(input_array: xr.DataArray) -> xr.DataArray:
    """input xarray.DataArray with esa cci classes and output dataarray with new classes
    but same coords and attributes as input"""
    new_class_1 = np.where(
        (input_array == 10)
        | (input_array == 11)
        | (input_array == 12)
        | (input_array == 20)
        | (input_array == 30)
        | (input_array == 40),
        1,
        input_array,
    )
    new_class_2 = np.where(
        (input_array == 60)
        | (input_array == 61)
        | (input_array == 62)
        | (input_array == 80)
        | (input_array == 81)
        | (input_array == 82)
        | (input_array == 90)
        | (input_array == 100),
        2,
        new_class_1,
    )
    new_class_3 = np.where(
        (input_array == 50)
        | (input_array == 70)
        | (input_array == 71)
        | (input_array == 72),
        3,
        new_class_2,
    )
    new_class_4 = np.where((input_array == 160) | (input_array == 170), 4, new_class_3)
    new_class_5 = np.where((input_array == 110) | (input_array == 130), 5, new_class_4)
    new_class_6 = np.where(
        (input_array == 120) | (input_array == 121) | (input_array == 122),
        6,
        new_class_5,
    )
    new_class_7 = np.where((input_array == 190), 7, new_class_6)
    new_class_8 = np.where(
        (input_array == 140)
        | (input_array == 150)
        | (input_array == 152)
        | (input_array == 153)
        | (input_array == 200)
        | (input_array == 201)
        | (input_array == 202)
        | (input_array == 220),
        8,
        new_class_7,
    )
    new_class_9 = np.where((input_array == 210), 9, new_class_8)
    new_class_10 = np.where((input_array == 180), 10, new_class_9)

    output_array_final = xr.DataArray(
        data=new_class_10, coords=input_array.coords, attrs=input_array.attrs
    )

    return output_array_final


def eunis_to_superclasses(input_array: xr.DataArray) -> xr.DataArray:
    """input xarray.DataArray with eunis classes and output dataarray with super classes
    but same coords and attributes as input"""

    new_class_1 = np.where(
        (input_array == 21)
        | (input_array == 22)
        | (input_array == 23)
        | (input_array == 24)
        | (input_array == 29)
        | (input_array == 30)
        | (input_array == 31)
        | (input_array == 33),
        1,
        input_array,
    )
    new_class_5 = np.where((input_array == 38) | (input_array == 43), 5, new_class_1)
    new_class_6 = np.where(
        (input_array == 35) | (input_array == 36) | (input_array == 37), 6, new_class_5
    )
    new_class_7 = np.where(
        (input_array == 25)
        | (input_array == 26)
        | (input_array == 27)
        | (input_array == 28),
        7,
        new_class_6,
    )
    new_class_8 = np.where((input_array == 1), 8, new_class_7)
    new_class_9 = np.where((input_array == 5), 9, new_class_8)
    new_class_10 = np.where((input_array == 6), 10, new_class_9)
    new_class_11 = np.where((input_array == 7) | (input_array == 8), 11, new_class_10)
    new_class_12 = np.where((input_array == 9) | (input_array == 11), 12, new_class_11)
    new_class_13 = np.where(
        (input_array == 16)
        | (input_array == 17)
        | (input_array == 18)
        | (input_array == 32),
        13,
        new_class_12,
    )
    new_class_14 = np.where((input_array == 19) | (input_array == 34), 14, new_class_13)
    new_class_15 = np.where((input_array == 39) | (input_array == 40), 15, new_class_14)
    new_class_16 = np.where((input_array == 41), 16, new_class_15)
    new_class_17 = np.where((input_array == 42), 17, new_class_16)
    new_class_18 = np.where(
        (input_array == 12)
        | (input_array == 13)
        | (input_array == 14)
        | (input_array == 15)
        | (input_array == 20),
        18,
        new_class_17,
    )
    new_class_19 = np.where((input_array == 10), 19, new_class_18)

    output_array_final = xr.DataArray(
        data=new_class_19, coords=input_array.coords, attrs=input_array.attrs
    )

    return output_array_final
