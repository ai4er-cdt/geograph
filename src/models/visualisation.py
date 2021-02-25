"""
This module contains visualisation functions for GeoGraphs.
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Callable

import folium
import geopandas as gpd
import networkx as nx
import shapely.geometry
from src.constants import CHERNOBYL_COORDS_WGS84, UTM35N, CEZ_DATA_PATH
from src.models import geograph


class GeoGraphViewer:
    """Class for viewing GeoGraph object"""

    def __init__(self, map_type: str = "folium") -> None:
        """Class for viewing GeoGraph object.

        Args:
            map_type (str, optional): defines which package is used for creating the
                map. One of ["folium","ipyleaflet"]. Folium is more widely compatible
                and may also run on google colab but the viewer functionality is
                limited, whereas ipyleaflet won't run on google colab but offers the
                full viewer functionality.
                Defaults to "folium".
        """
        if map_type == "ipyleaflet":
            raise NotImplementedError
        self.map_type = map_type
        self.widget = None

    def _repr_html_(self) -> str:
        """Return raw html of widget as string.

        This method gets called by IPython.display.display().
        """

        if self.widget is not None:
            return self.widget._repr_html_()  # pylint: disable=protected-access

    def add_graph(self, graph: geograph.GeoGraph) -> None:
        """Add graph to viewer.

        The added graph is visualised in the viewer.

        Args:
            graph (geograph.GeoGraph): GeoGraph to be shown.
        """

        if self.map_type == "folium":
            self._add_graph_to_folium_map(graph)

    def add_layer_control(self) -> None:
        """Add layer control to the viewer."""
        if self.map_type == "folium":
            folium.LayerControl().add_to(self.widget)

    def _add_graph_to_folium_map(self, graph: geograph.GeoGraph) -> None:
        """Add graph to folium map.

        Args:
            graph (geograph.GeoGraph): GeoGraph to be added.
        """
        self.widget = add_graph_to_folium_map(folium_map=self.widget, graph=graph.graph)


def add_graph_to_folium_map(
    folium_map: folium.Map = None,
    polygon_gdf: gpd.GeoDataFrame = None,
    color_column: str = "index",
    graph: Optional[nx.Graph] = None,
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
        graph (Optional[nx.Graph], optional): graph to be plotted. Defaults to None.
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
        node_gdf, edge_gdf = create_node_edge_geometries(graph, crs=crs)

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


def create_node_edge_geometries(
    graph: nx.Graph, crs: str = UTM35N
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create node and edge geometries for the networkx graph G.

    Returns node and edge geometries in two GeoDataFrames. The output can be used for
    plotting a graph.

    Args:
        graph (nx.Graph): graph with nodes and edges
        crs (str, optional): coordinate reference system. Defaults to UTM35N.

    Returns:
        Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: dataframes of nodes and edges
            respectively.
    """

    node_gdf = gpd.GeoDataFrame(columns=["id", "geometry"])
    rep_points = graph.nodes(data="rep_point")
    for idx, rep_point in rep_points:
        node_gdf.loc[idx] = [idx, rep_point]

    edge_gdf = gpd.GeoDataFrame(columns=["id", "geometry"])
    for idx, (node_a, node_b) in enumerate(graph.edges()):
        point_a = rep_points[node_a]
        point_b = rep_points[node_b]
        line = shapely.geometry.LineString([point_a, point_b])

        edge_gdf.loc[idx] = [idx, line]

    node_gdf = node_gdf.set_crs(crs)
    edge_gdf = edge_gdf.set_crs(crs)

    return node_gdf, edge_gdf


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
    exclusion_json_path: Optional[str] = CEZ_DATA_PATH,
    add_layer_control: bool = False,
) -> folium.Map:
    """Add polygons of the Chernobyl Exclusion Zone (CEZ) to a folium map.

    Args:
        folium_map (folium.Map): [description]
        exclusion_json_path (Optional[str], optional): path to the json file containing
            the CEZ polygons. Defaults to CEZ_DATA_PATH which requires access to
            the Jasmin servers and relevant shared workspaces.
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
