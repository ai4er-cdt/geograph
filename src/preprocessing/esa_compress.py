"""
esa-comp.py
================================================================
# from src.preprocessing.esa_comp import compress_esa, decompress_esa, FORW_D, REV_D

"""
import numpy as np


def _make_esa_map_d():
    """
    This function creats the mapping between the esa cci habitat labels and
    a reduced set of the same length. Only usable if the same habitat labels as in the
    Chernobyl region are used.
    :return: forw_d, rev_d; two dictionaries for the forwards / reverse transformation.
    """
    a= [0, 10, 11, 30, 40, 60, 61, 70, 80, 90, 100,
        110, 130, 150, 160, 180, 190, 200, 201, 210]
    b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19]
    forw_d = {}
    rev_d = {}
    for i in range(len(a)):
        forw_d[a[i]] = b[i]
        rev_d[b[i]] = a[i]
    return forw_d, rev_d


FORW_D, REV_D = _make_esa_map_d()  # Makes global objects for the mapping dicts.


# Create the unvectorised functions.

def _compress_esa(x):
    return FORW_D[x]


def _decompress_esa(x):
    return REV_D[x]


# Vectorize the functions

compress_esa = np.vectorize(_compress_esa)
decompress_esa = np.vectorize(_decompress_esa)
