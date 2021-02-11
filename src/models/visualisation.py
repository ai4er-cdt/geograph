"""
This file contains visualisation functions for vector, raster and graph data.
"""

import folium
import geopandas as gpd
import shapely.geometry
from src.constants import CHERNOBYL_COORDS_WGS84, UTM35N, GWS_DATA_DIR


def create_folium_map(
    m=None,
    polygon_gdf=None,
    color_column="index",
    graph=None,
    name="data",
    folium_tile_list=None,
    location=CHERNOBYL_COORDS_WGS84,
    crs=UTM35N,
    add_layer_control=False,
):

    if m is None:
        m = folium.Map(location, zoom_start=8, tiles=folium_tile_list.pop(0))

    if folium_tile_list is None:
        folium_tile_list = ["OpenStreetMap"]

    # Adding standard folium raster tiles
    for tiles in folium_tile_list:
        # special esri satellite data case
        if tiles == "esri":
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="esri",
                name="esri satellite",
                overlay=False,
                control=True,
            ).add_to(m)
        else:
            folium.TileLayer(tiles=tiles).add_to(m)

    # Adding polygon data
    if polygon_gdf is not None:
        # creating a color index that maps each category
        # in the color_column to an integer
        polygon_gdf["index"] = polygon_gdf.index
        polygon_gdf["color_index"] = (
            polygon_gdf[color_column].astype("category").cat.codes.astype("int64")
        )

        choropleth = folium.Choropleth(
            polygon_gdf,
            data=polygon_gdf,
            key_on="feature.properties.index",
            columns=["index", "color_index"],
            fill_color="YlOrBr",
            name=name + "_polygons",
        )
        choropleth.add_to(m)

        # adding popup markers with class name
        folium.features.GeoJsonPopup(fields=[color_column], labels=True).add_to(
            choropleth.geojson
        )

    # Adding graph data
    if graph is not None:
        node_gdf, edge_gdf = create_vis_gdfs_from_graph(graph, crs=crs)

        # add graph edges to map
        if not edge_gdf.empty:
            edges = folium.features.GeoJson(
                edge_gdf,
                name=name + "_graph_edges",
                style_function=get_style_function("#dd0000"),
            )
            edges.add_to(m)

        # add graph nodes/vertices to map
        node_marker = folium.vector_layers.Circle(radius=100, color="black")
        nodes = folium.features.GeoJson(
            node_gdf, marker=node_marker, name=name + "_graph_vetrices"
        )
        nodes.add_to(m)

    if add_layer_control:
        folium.LayerControl().add_to(m)

    return m


def create_vis_gdfs_from_graph(input_graph, crs=UTM35N):
    G = input_graph
    node_gdf = gpd.GeoDataFrame(columns=["id", "geometry"])
    rep_points = G.nodes(data="representative_point")
    for idx, rep_point in rep_points:
        node_gdf.loc[idx] = [idx, rep_point]

    edge_gdf = gpd.GeoDataFrame(columns=["id", "geometry"])
    for idx, (node_a, node_b) in enumerate(G.edges()):
        point_a = rep_points[node_a]
        point_b = rep_points[node_b]
        line = shapely.geometry.LineString([point_a, point_b])

        edge_gdf.loc[idx] = [idx, line]

    node_gdf = node_gdf.set_crs(crs)
    edge_gdf = edge_gdf.set_crs(crs)

    return node_gdf, edge_gdf


def get_style_function(color="#ff0000"):
    return lambda x: {"fillColor": color, "color": color}


def add_CEZ_to_map(m, add_layer_control=False):
    exclusion_json_path = GWS_DATA_DIR / "chernobyl_exclusion_zone_v1.geojson"
    exc_data = gpd.read_file(exclusion_json_path)

    colors = ["#808080", "#ffff99", "#ff9933", "#990000", "#ff0000", "#000000"]

    for index, row in exc_data.iterrows():
        folium.GeoJson(
            row["geometry"],
            name=row["name"],
            style_function=get_style_function(colors[index]),
        ).add_to(m)

    if add_layer_control:
        folium.LayerControl().add_to(m)

    return m
