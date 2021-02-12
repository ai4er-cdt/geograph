"""
This module provides helper functions to load land cover data close to chernobyl.
"""

import geopandas as gpd
from src.constants import GWS_DATA_DIR


def get_polesia_data(variant: str = "biotope") -> gpd.GeoDataFrame:
    """Return Polesia landcover data from shared workspace on Jasmin.
    
    Note that this requires access to Jasmin and the relevant shared workspaces.

    Args:
        variant (str, optional): Which variant of  the data to load. Must be 
            either "biotope" or "vegetation". Defaults to "biotope".

    Returns:
        gpd.GeoDataFrame: shape file data loaded as polygons in data frame
    """

    if variant == "biotope":
        data_path = GWS_DATA_DIR / "chernobyl_habitat_data" / "Biotope_EUNIS_ver1.shp"
    elif variant == "vegetation":
        data_path = GWS_DATA_DIR / "chernobyl_habitat_data" / "Vegetation_mape.shp"
    else:
        raise ValueError(
            "The variant does not match any existing Polesia data variant."
        )

    return gpd.read_file(data_path)
