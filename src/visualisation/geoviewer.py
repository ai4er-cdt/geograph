"""This module contains the GeoGraphViewer to visualise GeoGraphs"""

from __future__ import annotations
from typing import List, Dict, Union

import folium
import pandas as pd
import ipywidgets as widgets
import ipyleaflet
import traitlets
import logging

from src.constants import CHERNOBYL_COORDS_WGS84, WGS84
from src.models import geograph, metrics
from src.visualisation import folium_utils, graph_utils, widget_utils


class GeoGraphViewer(ipyleaflet.Map):
    """Class for interactively viewing a GeoGraph."""

    def __init__(
        self,
        center: List[int, int] = CHERNOBYL_COORDS_WGS84,
        zoom: int = 7,
        crs: Dict = ipyleaflet.projections.EPSG3857,
        layout: Union[widgets.Layout, None] = None,
        logging_level=logging.DEBUG,
        **kwargs
    ) -> None:
        """Class for interactively viewing a GeoGraph.

        Args:
            center ([type], optional): center of the map. Defaults to
                CHERNOBYL_COORDS_WGS84.
            zoom (int, optional): [description]. Defaults to 7.
            crs ([type], optional): [description]. Defaults to
                ipyleaflet.projections.EPSG3857.
            layout (widgets.Layout, optional): layout passed to ipyleaflet.Map
            logging_level: level of logs to be collected by self.logger. Defaults to
                logging.DEBUG.
        """
        super().__init__(
            center=center,
            zoom=zoom,
            scroll_wheel_zoom=True,
            crs=crs,
            zoom_snap=0.1,
            **kwargs
        )

        # Setting log with handler, that allows access to log via handler.show_logs()
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging_level)
        self.log_handler = widget_utils.OutputWidgetHandler()
        self.logger.addHandler(self.log_handler)

        if layout is None:
            self.layout = widgets.Layout(height="700px")

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
            style={"color": "black", "fillColor": "orange"},
            hover_style={"fillColor": "red", "fillOpacity": 0.2},
            point_style={
                "radius": 10,
                "color": "red",
                "fillOpacity": 0.8,
                "weight": 3,
            },
        )

        self.metrics = [
            "num_components",
            "avg_patch_area",
            "total_area",
            "avg_component_area",
            "avg_component_isolation",
        ]

        self.hover_widget = None
        self._widget_output = {}

        # Setting the order and current view of graph and map
        # with the latter two as traits
        self.add_traits(
            current_graph=traitlets.Unicode().tag(sync=True),
            current_map=traitlets.Unicode().tag(sync=True),
        )
        self.current_graph = ""
        self.current_map = "Map"  # set to the default map added above

        self.logger.info("Viewer successfully setup.")

    def set_layer_visibility(
        self, layer_type: str, layer_name: str, layer_subtype: str, active: bool
    ) -> None:
        """Set visiblity for layer_dict[layer_type][layer_name][layer_subtype]."""
        print(layer_type, layer_name, layer_subtype)
        self.layer_dict[layer_type][layer_name][layer_subtype]["active"] = active

    def hidde_all_layers(self) -> None:
        """ Hide all layers in self.layer_dict."""
        for layer_type, type_dict in self.layer_dict.items():
            for layer_name in type_dict:
                if layer_type == "maps":
                    self.set_layer_visibility(layer_type, layer_name, "map", False)
                elif layer_type == "graphs":
                    self.set_layer_visibility(layer_type, layer_name, "graph", False)
                    self.set_layer_visibility(layer_type, layer_name, "pgons", False)
                    self.set_layer_visibility(
                        layer_type, layer_name, "components", False
                    )
        self.layer_update()

    # @log_out.capture()
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

            nodes, edges = graph_utils.create_node_edge_geometries(nx_graph)
            graph_geometries = pd.concat([edges, nodes]).reset_index()
            graph_geo_data = ipyleaflet.GeoData(
                geo_dataframe=graph_geometries.to_crs(WGS84),
                name=graph_name + "_graph",
                **self.custom_style
            )

            pgon_df = graph.df.loc[list(nx_graph)]
            pgon_df["area_in_m2"] = pgon_df.area
            pgon_df = pgon_df.to_crs(WGS84)
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
                hover_style={"fillOpacity": 1},
                style={"fillOpacity": 0.5},
            )

            if self.hover_widget is None:
                # TODO: refactor this into separate methods that are called in init.

                hover_html = widgets.HTML("""Hover over patches""")
                hover_html.layout.margin = "10px 10px 10px 10px"
                hover_html.layout.max_width = "300px"

                def hover_callback(
                    feature, **kwargs
                ):  # pylint: disable=unused-argument
                    """Adapt text of `hover_html` widget to patch"""
                    new_value = """<b>Current Patch</b></br>
                        <b>Class label:</b> {}</br>

                        <b>Area:</b> {:.2f} m^2
                    """.format(
                        feature["properties"]["class_label"],
                        feature["properties"]["area_in_m2"],
                    )
                    hover_html.value = new_value  # pylint: disable=cell-var-from-loop

                self.hover_callback = hover_callback
                self.hover_widget = hover_html

            pgon_choropleth.on_hover(self.hover_callback)

            graph_metrics = []

            if not is_habitat:
                for metric in self.metrics:
                    graph_metrics.append(metrics.get_metric(metric, graph))

            self.layer_dict["graphs"][graph_name] = dict(
                is_habitat=is_habitat,
                graph=dict(layer=graph_geo_data, active=True),
                pgons=dict(layer=pgon_choropleth, active=True),
                components=dict(layer=None, active=False),
                metrics=graph_metrics,
            )

        self.current_graph = name
        self.layer_update()
        self.logger.info("Added graph.")

    def add_hover_widget(self) -> None:
        """Add hover widget for graph."""
        control = ipyleaflet.WidgetControl(
            widget=self.hover_widget, position="topright"
        )
        self.add_control(control)

    def layer_update(self) -> None:
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

    def set_graph_style(self, radius: float = 10, node_color=None) -> None:
        """Set the style of any graphs added to viewer.

        Args:
            radius (float): radius of nodes in graph. Defaults to 10.
        """

        for name, graph in self.layer_dict["graphs"].items():
            layer = graph["graph"]["layer"]

            # Below doesn't work because traitlet change not observed
            # layer.point_style['radius'] = radius

            self.custom_style["point_style"]["radius"] = radius
            self.custom_style["style"]["fillColor"] = node_color
            layer = ipyleaflet.GeoData(
                geo_dataframe=layer.geo_dataframe, name=layer.name, **self.custom_style
            )
            self.layer_dict["graphs"][name]["graph"]["layer"] = layer
        self.layer_update()

    def add_settings_widget(self) -> None:
        """Add settings widget to viewer."""

        habitats_tab = self._create_habitat_tab()
        diff_tab = self._create_diff_tab()
        settings_tab = self._create_settings_tab()
        metrics_widget = self._create_metrics_widget()

        tab_nest_dict = dict(
            Layers=habitats_tab,
            Metrics=metrics_widget,
            Diff=diff_tab,
            Settings=settings_tab,
            Log=self.logger,
        )

        tab_nest = widgets.Tab()
        tab_nest.children = list(tab_nest_dict.values())
        for i, title in enumerate(tab_nest_dict):
            tab_nest.set_title(i, title)

        self.add_control(ipyleaflet.WidgetControl(widget=tab_nest, position="topright"))

        # self.add_control(ipyleaflet.LayersControl(position="topleft"))

        header = widgets.HTML(
            """<b>GeoGraph</b>""", layout=widgets.Layout(padding="3px 10px 3px 10px")
        )
        self.add_control(
            ipyleaflet.WidgetControl(widget=header, position="bottomright")
        )

    def add_widgets(self) -> None:
        """Add all widgets to viewer"""
        self.add_settings_widget()
        self.add_control(ipyleaflet.FullScreenControl())
        self.add_control(ipyleaflet.ScaleControl(position="bottomleft"))


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
        self.widget = folium_utils.add_graph_to_folium_map(
            folium_map=self.widget, graph=graph.graph
        )
