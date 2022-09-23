"""This file is for constants relevant to the binder demo."""
from geograph.constants import PROJECT_PATH


# Data directory on GWS
DATA_DIR = PROJECT_PATH / "data"
# Polygon data of Chernobyl Exclusion Zone (CEZ)
ROIS = DATA_DIR / "chernobyl" / "chernobyl_rois.geojson"

# Link to ESA CCI Land cover Legend
ESA_CCI_LEGEND_LINK = (
    "https://www.dropbox.com/s/bget0phawnahd8v/ESACCI-LC-Legend.csv?dl=1"
)
