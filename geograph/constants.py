"""All project wide constants are saved in this module."""
# Place all your constants here
import os
import pathlib

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))

# Coordinate reference systems (crs)
WGS84 = "EPSG:4326"  # WGS84 standard crs (latitude, longitude)
UTM35N = "EPSG:32635"  # https://epsg.io/32635 - preferred crs for chernobyl region

# Coordinates
CHERNOBYL_COORDS_WGS84 = (
    51.389167,
    30.099444,
)  # coordinates of chernobyl power reactor
CHERNOBYL_COORDS_UTM35N = (715639.1222290158, 5697662.734402668)
