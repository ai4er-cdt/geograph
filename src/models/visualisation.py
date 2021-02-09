"""
This file contains visualisation functions for vector, raster and graph data.
"""

import folium
import geopandas as gpd
import shapely.geometry
from src.constants import CHERNOBYL_COORDS_UTM35N, CHERNOBYL_COORDS_WGS84, UTM35N_CODE


def create_folium_map(polygon_gdf_list=[],
                      polygon_gdf_colour_col_list=[],
                      graph_list=[],
                      folium_tile_list=['OpenStreetMap'],
                      location=CHERNOBYL_COORDS_WGS84,
                      crs=UTM35N_CODE):

    m = folium.Map(location, zoom_start=8, tiles=folium_tile_list[0])

    # Adding standard folium raster tiles
    for tiles in folium_tile_list[1:]:
        # special esri satellite data case
        if tiles == 'esri':
            folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'esri',
            name = 'esri satellite',
            overlay = False,
            control = True
            ).add_to(m)
        else:
            folium.TileLayer(tiles=tiles).add_to(m)

    # Adding polygon data
    for poly_gdf, color_col in zip(polygon_gdf_list, polygon_gdf_colour_col_list):
        # creating a color index that maps each category
        # in the color_col column to an integer
        poly_gdf['color_index'] = poly_gdf[color_col].astype('category').cat.codes.astype('int64')
        poly_gdf['index'] = poly_gdf.index

        choropleth = folium.Choropleth(poly_gdf, data=poly_gdf, key_on='feature.properties.index',
                                   columns=['index','color_index'], fill_color='YlOrBr',
                                   name='data')
        choropleth.add_to(m)

        # adding popup markers with class name
        folium.features.GeoJsonPopup(fields=[color_col], labels=True ).add_to(choropleth.geojson)

    #Adding graph data
    for graph in graph_list:
        #TODO: implement graph plotting functionality
        node_gdf, edge_gdf = create_vis_gdfs_from_graph(graph, crs=crs)

        # add graph edges to map
        if not edge_gdf.empty:
            edges = folium.features.GeoJson(edge_gdf, name='graph_edges', style_function=get_style_function('#dd0000'))
            edges.add_to(m)

        # add graph nodes/vertices to map
        node_marker = folium.vector_layers.Circle(radius=100, color='black')
        nodes = folium.features.GeoJson(node_gdf,marker=node_marker, name='graph_vetrices')
        nodes.add_to(m)

    folium.LayerControl().add_to(m)

    return m


def create_vis_gdfs_from_graph(input_graph, circle_size=100, crs=UTM35N_CODE):
    G = input_graph
    node_gdf = gpd.GeoDataFrame(columns=['id', 'geometry'])
    rep_points = G.nodes(data='representative_point')
    for idx, rep_point in rep_points:
        node_gdf.loc[idx] = [idx, rep_point]

    edge_gdf = gpd.GeoDataFrame(columns=['id', 'geometry'])
    for idx, (node_a, node_b) in enumerate(G.edges()):
        point_a = rep_points[node_a]
        point_b = rep_points[node_b]
        line = shapely.geometry.LineString([point_a, point_b])

        edge_gdf.loc[idx] = [idx, line]

    node_gdf = node_gdf.set_crs(crs)
    edge_gdf = edge_gdf.set_crs(crs)

    return node_gdf, edge_gdf

def get_style_function(color = '#ff0000'):
    return lambda x: {'fillColor': color, 'color': color}