"""
This file is used to save all project wide constants such as the path of the
source folder, the project path, etc.
"""

# Place all your constants here
import os
import pathlib
from pyproj import Transformer

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)

# Jasmine data dir
GWS_DATA_DIR = pathlib.Path("/gws/nopw/j04/ai4er/guided-team-challenge/2021/biodiversity")

# Coordinate reference systems and locations
WGS84_CODE = 'EPSG:4326'
UTM35N_CODE = 'EPSG:32635' # coordinate system of chernobyl data
transformer = Transformer.from_crs(WGS84_CODE, UTM35N_CODE)
CHERNOBYL_COORDS_WGS84 = (51.389167, 30.099444) # coordinates of chernobyl power reactor
CHERNOBYL_COORDS_UTM35N = transformer.transform(*CHERNOBYL_COORDS_WGS84)
