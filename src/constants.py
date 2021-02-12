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
GWS_DATA_DIR = pathlib.Path(
    "/gws/nopw/j04/ai4er/guided-team-challenge/2021/biodiversity"
)

# Polygon data of Chernobyl Exclusion Zone (CEZ)
CEZ_DATA_PATH = GWS_DATA_DIR / "chernobyl_exclusion_zone_v1.geojson"

# CEDA directory
CEDA_DIR = pathlib.Path("/neodc")
# ESA Landcover directory in CEDA
ESA_LANDCOVER_DIR = CEDA_DIR / "esacci/land_cover/data/land_cover_maps/v2.0.7"

# Coordinate reference systems (crs)
WGS84 = "EPSG:4326"  # WGS84 standard crs (latitude, longitude)
UTM35N = "EPSG:32635"  # https://epsg.io/32635 - preferred crs for chernobyl region
UCS2000_TM10 = "EPSG:6384"  # https://epsg.io/6384 - reference system for ukaraine
PREFERRED_CRS = UTM35N  # backwards compatability - preferred crs for chernobyl region

# Coordinates
CHERNOBYL_COORDS_WGS84 = (
    51.389167,
    30.099444,
)  # coordinates of chernobyl power reactor
CHERNOBYL_COORDS_UTM35N = (715639.1222290158, 5697662.734402668)
