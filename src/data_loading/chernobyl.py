"""
This module provides helper functions to load land cover data close to chernobyl.
"""

import geopandas as gpd
from src.constants import GWS_DATA_DIR

def get_bio_data():
    # Getting biotope data
    bio_path = GWS_DATA_DIR / "chernobyl_habitat_data" / "Biotope_EUNIS_ver1.shp"
    return gpd.read_file(bio_path)

def get_veg_data():
    # Getting vegetation data
    veg_path = GWS_DATA_DIR / "chernobyl_habitat_data" / "Vegetation_mape.shp"
    return gpd.read_file(veg_path)