"""Test module"""
from src.constants import GWS_DATA_DIR, PREFERRED_CRS
from src.models.geograph import GeoGraph

shape_pth = GWS_DATA_DIR / "chernobyl_habitat_data" / "Vegetation_mape.shp"
raster_pth = GWS_DATA_DIR / "esa_cci_rois" / "esa_cci_2015_chernobyl.tif"
# chernobyl_graph = GeoGraph(data=raster_pth)
# chernobyl_graph = GeoGraph(data=PROJECT_PATH / "data" / "geo_data.gpkg")
chernobyl_graph = GeoGraph(data=shape_pth, crs=PREFERRED_CRS)

# Diagnostics
print(chernobyl_graph.graph.number_of_nodes(), chernobyl_graph.graph.number_of_edges())
print(list(chernobyl_graph.graph.adj[0]))
chernobyl_graph.graph.add_edge(0, 5)
print(list(chernobyl_graph.graph.adj[0]))
print(chernobyl_graph.df.head())
chernobyl_graph.df = chernobyl_graph.df.drop([0, 1])
print(chernobyl_graph.df.head())
