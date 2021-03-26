"""This file is for constants relevant to the binder demo."""

from geograph.constants import PROJECT_PATH

# Data directory on GWS
DATA_DIR = PROJECT_PATH / "data"
# Polygon data of Chernobyl Exclusion Zone (CEZ)
ROIS = DATA_DIR / "chernobyl" / "chernobyl_rois.geojson"
