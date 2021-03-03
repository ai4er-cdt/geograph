"""
This module contains visualisation functions for GeoGraphs.
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Callable, Dict

import folium
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely.geometry
import ipywidgets as widgets
import ipyleaflet
import markdown
import traitlets


from src.constants import CHERNOBYL_COORDS_WGS84, WGS84, UTM35N, CEZ_DATA_PATH
from src.models import geograph


class GeoGraphViewer(ipyleaflet.Map):
    """Class for interactively viewing a GeoGraph."""

    log = widgets.Output(layout={"border": "1px solid black"})

    def __init__(
        self,
        center: List[int, int] = CHERNOBYL_COORDS_WGS84,
        zoom: int = 7,
        crs: Dict = ipyleaflet.projections.EPSG3857,
        **kwargs
    ) -> None:
        """Class for interactively viewing a GeoGraph.

        Args:
            center ([type], optional): center of the map. Defaults to
                CHERNOBYL_COORDS_WGS84.
            zoom (int, optional): [description]. Defaults to 7.
            crs ([type], optional): [description]. Defaults to
                ipyleaflet.projections.EPSG3857.
        """
        super().__init__(
            center=center,
            zoom=zoom,
            scroll_wheel_zoom=True,
            crs=crs,
            zoom_snap=0.1,
            **kwargs
        )
        # Note: entries in layer dict follow the convention:
        # ipywidgets_layer = layer_dict[type][name][subtype]["layer"]
        # Layers of type "maps" only have subtype "map".
        self.layer_dict = dict(
            maps=dict(
                Map=dict(
                    map=dict(
                        layer=ipyleaflet.TileLayer(base=True, max_zoom=19, min_zoom=4),
                        active=True,
                    )
                )
            ),
            graphs=dict(),
        )
        self.graph_dict = dict()
        self.custom_style = dict(
            style={"color": "black", "fillColor": "#3366cc"},
            hover_style={"fillColor": "red", "fillOpacity": 0.2},
            point_style={
                "radius": 10,
                "color": "red",
                "fillOpacity": 0.8,
                "fillColor": "blue",
                "weight": 3,
            },
        )
        self._widget_output = {}

    @log.capture()
    def add_graph(self, graph: geograph.GeoGraph, name: str = "Graph") -> None:
        """Add GeoGraph to viewer.

        Args:
            graph (geograph.GeoGraph): graph to be added
            name (str, optional): name shown in control panel. Defaults to "Graph".
        """
        nx_graphs = {name: graph.graph}
        for habitat_name, habitat in graph.habitats.items():
            nx_graphs[habitat_name] = habitat.graph

        for idx, (graph_name, nx_graph) in enumerate(nx_graphs.items()):

            is_habitat = idx > 0

            nodes, edges = create_node_edge_geometries(nx_graph)
            graph_geometries = pd.concat([nodes, edges]).reset_index()
            graph_geo_data = ipyleaflet.GeoData(
                geo_dataframe=graph_geometries.to_crs(WGS84),
                name=graph_name + "_graph",
                **self.custom_style
            )

            pgon_df = graph.df.loc[list(nx_graph)].to_crs(WGS84)
            pgon_geo_data = pgon_df.__geo_interface__

            # ipyleaflet.choropleth can't reach individual properties for key
            for feature in pgon_geo_data["features"]:
                feature["class_label"] = feature["properties"]["class_label"]

            unique_classes = list(graph.df.class_label.unique())
            choro_data = dict(zip(unique_classes, range(len(unique_classes))))

            pgon_choropleth = ipyleaflet.Choropleth(
                geo_data=pgon_geo_data,
                choro_data=choro_data,
                key_on="class_label",
                border_color="black",
                style={"fillOpacity": 0.8},
            )

            self.layer_dict["graphs"][graph_name] = dict(
                is_habitat=is_habitat,
                graph=dict(layer=graph_geo_data, active=True),
                pgons=dict(layer=pgon_choropleth, active=True),
            )
        self._layer_update()

    # @log.capture()
    # def _add_nx_graph(self):

    @log.capture()
    def _layer_update(self) -> None:
        """Update `self.layer` tuple from `self.layer_dict`."""
        layers = [
            map_layer["map"]["layer"]
            for map_layer in self.layer_dict["maps"].values()
            if map_layer["map"]["active"]
        ]
        for graph in self.layer_dict["graphs"].values():
            if graph["pgons"]["active"]:
                layers.append(graph["pgons"]["layer"])
            if graph["graph"]["active"]:
                layers.append(graph["graph"]["layer"])

        self.layers = tuple(layers)

    @log.capture()
    def set_graph_style(self, radius: float = 10, node_color="blue") -> None:
        """Set the style of any graphs added to viewer.

        Args:
            radius (float): radius of nodes in graph. Defaults to 10.
        """

        for name, layer_dict in self.layer_dict["graphs"].items():
            layer = layer_dict["layer"]

            # Below doesn't work because traitlet change not observed
            # layer.point_style['radius'] = radius

            self.custom_style["point_style"]["radius"] = radius
            self.custom_style["style"]["fillColor"] = node_color
            layer = ipyleaflet.GeoData(
                geo_dataframe=layer.geo_dataframe, name=layer.name, **self.custom_style
            )
            self.layer_dict["graphs"][name]["layer"] = layer
        self._layer_update()

    @log.capture()
    def remove_graphs(
        self,
        button: widgets.widget_button.Button = None,  # pylint: disable=unused-argument
    ) -> None:
        """Temporarily remove all graphs.

        Args:
            button (widgets.widget_button.Button): button object that is passed by the
                observe method of a ipywidgets button. Is ignored. Defaults to None.
        """
        for layer in self.layers:
            if layer.name[0:5] == "Graph":
                self.remove_layer(layer)

    @log.capture()
    def _switch_layer_visibility(self, change: Dict):
        """Switch layer visibility according to change.

        Args:
            change (Dict): change dict provided by checkbox widget
        """
        if change["name"] == "value":
            layer_type = change["owner"].layer_type
            layer_name = change["owner"].layer_name
            layer_subtype = change["owner"].layer_subtype
            self.layer_dict[layer_type][layer_name][layer_subtype]["active"] = change[
                "new"
            ]

        self._layer_update()

    @log.capture()
    def _create_habitat_tab(self) -> widgets.VBox:
        """Create tab widget for habitats.

        Returns:
            widgets.VBox: widget
        """
        checkboxes = []
        graphs = [
            (name, "graphs", layer_subtype, graph)
            for name, graph in self.layer_dict["graphs"].items()
            for layer_subtype in ["graph", "pgons"]
        ]
        maps = [
            (name, "maps", "map", map_layer["map"])
            for name, map_layer in self.layer_dict["maps"].items()
        ]
        for layer_name, layer_type, layer_subtype, layer_dict in maps + graphs:

            layout = widgets.Layout(padding="0px 0px 0px 0px")
            # indenting habitat checkboxes
            if layer_type == "graphs":
                if layer_dict["is_habitat"]:
                    layout = widgets.Layout(padding="0px 0px 0px 25px")

            checkbox = widgets.Checkbox(
                value=True,
                description="_".join([layer_name, layer_subtype]),
                disabled=False,
                indent=False,
                layout=layout,
            )
            checkbox.add_traits(
                layer_type=traitlets.Unicode().tag(sync=True),
                layer_subtype=traitlets.Unicode().tag(sync=True),
                layer_name=traitlets.Unicode().tag(sync=True),
            )
            checkbox.layer_type = layer_type
            checkbox.layer_name = layer_name
            checkbox.layer_subtype = layer_subtype

            checkbox.observe(self._switch_layer_visibility)
            checkboxes.append(checkbox)

        habitat_tab = widgets.VBox(checkboxes)

        return habitat_tab

    @log.capture()
    def _create_diff_tab(self) -> widgets.VBox:
        """Create tab widget for diff.

        Returns:
            widgets.VBox: widget
        """

        time_slider1 = widgets.IntSlider(
            min=1960, max=2021, step=1, value=1990, description="Start time:"
        )
        time_slider2 = widgets.IntSlider(
            min=1960, max=2021, step=1, value=2010, description="End time:"
        )

        compute_node_button = widgets.Button(
            description="Compute node diff",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip=(
                "This computes the differences between of the nodes in the graph"
                " at start time and the graph at end time."
            ),
            icon="",  # (FontAwesome names without the `fa-` prefix)
        )

        compute_pgon_button = widgets.Button(
            description="Compute polygon diff",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip=(
                "This computes the differences between of the polygons in the"
                " graph at start time and the graph at end time."
            ),
            icon="",  # (FontAwesome names without the `fa-` prefix)
        )

        diff_tab = widgets.VBox(
            [time_slider1, time_slider2, compute_node_button, compute_pgon_button]
        )

        return diff_tab

    @log.capture()
    def _create_settings_tab(self) -> widgets.VBox:
        """Create tab widget for settings.

        Returns:
            widgets.VBox: tab widget
        """

        radius_slider = widgets.FloatSlider(
            min=0.01, max=100.0, step=0.005, value=5.0, description="Node radius:"
        )

        node_color_picker = widgets.ColorPicker(
            concise=True, description="Node color", value="blue", disabled=False
        )

        self._widget_output["settings_tab"] = widgets.interactive_output(
            self.set_graph_style,
            dict(radius=radius_slider, node_color=node_color_picker),
        )

        zoom_slider = widgets.FloatSlider(
            description="Zoom level:", min=0, max=15, value=7
        )
        widgets.jslink((zoom_slider, "value"), (self, "zoom"))

        settings_tab = widgets.VBox(
            [
                zoom_slider,
                node_color_picker,
                radius_slider,
            ]
        )

        return settings_tab

    @log.capture()
    def _create_metrics_widget(self) -> widgets.VBox:
        """Create metrics visualisation widget.

        Returns:
            widgets.VBox: metrics widget
        """

        widget = widgets.VBox(
            [widgets.HTML(markdown.markdown("""&nbsp;&nbsp;Metrics&nbsp;&nbsp;"""))]
        )
        return widget

    @log.capture()
    def add_settings_widget(self) -> None:
        """Add settings widget to viewer."""

        habitats_tab = self._create_habitat_tab()
        diff_tab = self._create_diff_tab()
        settings_tab = self._create_settings_tab()

        tab_nest_dict = dict(
            Layers=habitats_tab,
            Diff=diff_tab,
            Settings=settings_tab,
            Log=self.log,
        )

        tab_nest = widgets.Tab()
        tab_nest.children = list(tab_nest_dict.values())
        for i, title in enumerate(tab_nest_dict):
            tab_nest.set_title(i, title)

        self.add_control(ipyleaflet.WidgetControl(widget=tab_nest, position="topright"))

        metrics_widget = self._create_metrics_widget()
        self.add_control(
            ipyleaflet.WidgetControl(widget=metrics_widget, position="topright")
        )

        # self.add_control(ipyleaflet.LayersControl(position="topleft"))

        header = widgets.HTML(markdown.markdown("""&nbsp;&nbsp;GeoGraph&nbsp;&nbsp;"""))
        self.add_control(
            ipyleaflet.WidgetControl(widget=header, position="bottomright")
        )

    @log.capture()
    def add_widgets(self) -> None:
        """Add all widgets to viewer"""
        self.add_settings_widget()
        self.add_control(ipyleaflet.FullScreenControl())


class FoliumGeoGraphViewer:
    """Class for viewing GeoGraph object without ipywidgets"""

    def __init__(self) -> None:
        """Class for viewing GeoGraph object without ipywidgets."""
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

        self._add_graph_to_folium_map(graph)

    def add_layer_control(self) -> None:
        """Add layer control to the viewer."""
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
