"""
This file contains visualisation functions for vector, raster and graph data.
"""

import folium


def create_folium_map(polygon_df_list=[],
                      polygon_df_colour_col_list=[],
                      folium_tile_list=['OpenStreetMap'],
                      location=[51.386998452, 30.092666296]):

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
    for poly_df, color_col in zip(polygon_df_list, polygon_df_colour_col_list):
        # creating a color index that maps each category
        # in the color_col column to an integer
        poly_df['color_index'] = poly_df[color_col].astype('category').cat.codes.astype('int64')
        poly_df['index'] = poly_df.index

        choropleth = folium.Choropleth(poly_df, data=poly_df, key_on='feature.properties.index',
                                   columns=['index','color_index'], fill_color='YlOrBr',
                                   name='data')
        choropleth.add_to(m)

        # adding popup markers with class name
        folium.features.GeoJsonPopup(fields=[color_col], labels=True ).add_to(choropleth.geojson)

    folium.LayerControl().add_to(m)

    return m
