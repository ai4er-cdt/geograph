"""This file is for constants relevant to the binder demo."""

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
