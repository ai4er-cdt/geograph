"""
This file is used to save all project wide constants such as the path of the
source folder, the project path, etc.
"""

# Place all your constants here
import os
import pathlib

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)

# Data directory on GWS
GWS_DATA_DIR = pathlib.Path(
    "/gws/nopw/j04/ai4er/guided-team-challenge/2021/biodiversity"
)
# CEDA directory
CEDA_DIR = pathlib.Path("/neodc")
# ESA Landcover directory in CEDA
ESA_LANDCOVER_DIR = CEDA_DIR / "esacci/land_cover/data/land_cover_maps/v2.0.7"

# Coordinate reference systems (crs)
WGS84 = "EPSG:4326"  # WGS84 standard crs (latitude, longitude)
PREFERRED_CRS = "EPSG:32635"  # UTM 35 - preferred crs for chernobyl region
