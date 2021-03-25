"""Module with utility functions to plot graphs in folium."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import folium
import geograph
import geopandas as gpd
from geograph.constants import CHERNOBYL_COORDS_WGS84, UTM35N
from geograph.visualisation import graph_utils


def add_graph_to_folium_map(
    folium_map: folium.Map = None,
    polygon_gdf: gpd.GeoDataFrame = None,
    color_column: str = "index",
    graph: Optional[geograph.GeoGraph] = None,
    name: str = "data",
    folium_tile_list: Optional[List[str]] = None,
    location: Tuple[float, float] = CHERNOBYL_COORDS_WGS84,
    crs: str = UTM35N,
    add_layer_control: bool = False,
) -> folium.Map:
    """Create a visualisation map of the given polygons and `graph` in folium.

    The polygons in `polygon_gdf` and `graph` are displayed on a folum map.
    It is intended that the graph was build from `polygon_gdf`, but it is not required.
    If given `map`, it will be put on this existing folium map.

    Args:
        folium_map (folium.Map, optional): map to add polygons and graph to.
            Defaults to None.
        polygon_gdf (gpd.GeoDataFrame, optional): data containing polygon.
            Defaults to None.
        color_column (str, optional): column in polygon_gdf that determines which color
            is given to each polygon. Can be categorical values. Defaults to "index".
        graph (Optional[geograph.GeoGraph], optional): graph to be plotted.
            Defaults to None.
        name (str, optional): prefix to all the folium layer names shown in layer
            control of map (if added). Defaults to "data".
        folium_tile_list (Optional[List[str]], optional): list of folium.Map tiles to be
            add to the map. See folium.Map docs for options. Defaults to None.
        location (Tuple[float, float], optional): starting location in WGS84 coordinates
            Defaults to CHERNOBYL_COORDS_WGS84.
        crs (str, optional): coordinates reference system to be used.
            Defaults to UTM35N.
        add_layer_control (bool, optional): whether to add layer controls to map.
            Warning: only use this when you don't intend to add any additional data
            after calling this function to the map. May cause bugs otherwise.
            Defaults to False.

    Returns:
        folium.Map: map with polygons and graph displayed as described
    """

    if folium_tile_list is None:
        folium_tile_list = ["OpenStreetMap"]

    if folium_map is None:
        folium_map = folium.Map(location, zoom_start=8, tiles=folium_tile_list.pop(0))

    # Adding standard folium raster tiles
    for tiles in folium_tile_list:
        # special esri satellite data case
        if tiles == "esri":
            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/ArcGIS/rest/"
                    "services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ),
                attr="esri",
                name="esri satellite",
                overlay=False,
                control=True,
            ).add_to(folium_map)
        else:
            folium.TileLayer(tiles=tiles).add_to(folium_map)

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
        choropleth = remove_choropleth_color_legend(choropleth)
        choropleth.add_to(folium_map)

        # adding popup markers with class name
        folium.features.GeoJsonPopup(fields=[color_column], labels=True).add_to(
            choropleth.geojson
        )

    # Adding graph data
    if graph is not None:
        node_gdf, edge_gdf = graph_utils.create_node_edge_geometries(graph, crs=crs)

        # add graph edges to map
        if not edge_gdf.empty:
            edges = folium.features.GeoJson(
                edge_gdf,
                name=name + "_graph_edges",
                style_function=get_style_function("#dd0000"),
            )
            edges.add_to(folium_map)

        # add graph nodes/vertices to map
        node_marker = folium.vector_layers.Circle(radius=100, color="black")
        nodes = folium.features.GeoJson(
            node_gdf, marker=node_marker, name=name + "_graph_vetrices"
        )
        nodes.add_to(folium_map)

    if add_layer_control:
        folium.LayerControl().add_to(folium_map)

    return folium_map


def get_style_function(color: str = "#ff0000") -> Callable[[], dict]:
    """Return lambda function that returns a dict with the `color` given.

    The returned lambda function can be used as a style function for folium.

    Args:
        color (str, optional): color to be used in dict. Defaults to "#ff0000".

    Returns:
        Callable[[], dict]: style function
    """

    return lambda x: {"fillColor": color, "color": color}


def add_cez_to_map(
    folium_map: folium.Map,
    exclusion_json_path: Optional[str] = None,
    add_layer_control: bool = False,
) -> folium.Map:
    """Add polygons of the Chernobyl Exclusion Zone (CEZ) to a folium map.

    Args:
        folium_map (folium.Map): [description]
        exclusion_json_path (Optional[str], optional): path to the json file containing
            the CEZ polygons. Defaults to None.
        add_layer_control (bool, optional): whether to add layer controls to map.
            Warning: only use this when you don't intend to add any additional data
            after calling this function to the map. May cause bugs otherwise.
            Defaults to False.
    Returns:
        folium.Map: map with CEZ polygons added
    """

    exc_data = gpd.read_file(exclusion_json_path)

    colors = ["#808080", "#ffff99", "#ff9933", "#990000", "#ff0000", "#000000"]

    for index, row in exc_data.iterrows():
        folium.GeoJson(
            row["geometry"],
            name=row["name"],
            style_function=get_style_function(colors[index]),
        ).add_to(folium_map)

    if add_layer_control:
        folium.LayerControl().add_to(folium_map)

    return folium_map


def remove_choropleth_color_legend(
    choropleth_map: folium.features.Choropleth,
) -> folium.features.Choropleth:
    """Remove color legend from Choropleth folium map.

    Solution proposed by `nhpackard` in the following GitHub issue in the folium repo:
    https://github.com/python-visualization/folium/issues/956

    Args:
        choropleth_map (folium.features.Choropleth): a Choropleth map

    Returns:
        folium.features.Choropleth: the same map without color legend
    """
    for key in choropleth_map._children:  # pylint: disable=protected-access
        if key.startswith("color_map"):
            del choropleth_map._children[key]  # pylint: disable=protected-access

    return choropleth_map
