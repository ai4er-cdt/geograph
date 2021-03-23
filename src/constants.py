"""
This file is used to save all project wide constants such as the path of the
source folder, the project path, etc.
"""

# Place all your constants here
import os
import pathlib

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))


# Data directory on GWS
DATA_DIR = PROJECT_PATH / "data"
# Polygon data of Chernobyl Exclusion Zone (CEZ)
ROIS = DATA_DIR / "chernobyl" / "chernobyl_rois.geojson"


# Coordinate reference systems (crs)
WGS84 = "EPSG:4326"  # WGS84 standard crs (latitude, longitude)
UTM35N = "EPSG:32635"  # https://epsg.io/32635 - preferred crs for chernobyl region
UCS2000_TM10 = "EPSG:6384"  # https://epsg.io/6384 - reference system for ukraine

# Coordinates
CHERNOBYL_COORDS_WGS84 = (
    51.389167,
    30.099444,
)  # coordinates of chernobyl power reactor
CHERNOBYL_COORDS_UTM35N = (715639.1222290158, 5697662.734402668)
