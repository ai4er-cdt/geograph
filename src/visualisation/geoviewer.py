"""This module contains the GeoGraphViewer to visualise GeoGraphs"""

from __future__ import annotations
from typing import List, Dict, Union

import folium
import pandas as pd
import ipywidgets as widgets
import ipyleaflet
import traitlets

from src.constants import CHERNOBYL_COORDS_WGS84, WGS84
from src.models import geograph, metrics
from src.visualisation import folium_utils, graph_utils


class GeoGraphViewer(ipyleaflet.Map):
    """Class for interactively viewing a GeoGraph."""

    log = widgets.Output(layout={"border": "1px solid black"})

    def __init__(
        self,
        center: List[int, int] = CHERNOBYL_COORDS_WGS84,
        zoom: int = 7,
        crs: Dict = ipyleaflet.projections.EPSG3857,
        layout: Union[widgets.Layout, None] = None,
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
        """
        super().__init__(
            center=center,
            zoom=zoom,
            scroll_wheel_zoom=True,
            crs=crs,
            zoom_snap=0.1,
            **kwargs
        )
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

    @log.capture()
    def set_layer_visibility(
        self, layer_type: str, layer_name: str, layer_subtype: str, active: bool
    ) -> None:
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

                @self.log.capture()
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
        self.layer_update()

    @log.capture()
    def add_hover_widget(self) -> None:
        """Add hover widget for graph."""
        control = ipyleaflet.WidgetControl(
            widget=self.hover_widget, position="topright"
        )
        self.add_control(control)

    @log.capture()
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

    @log.capture()
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

        self.layer_update()

    @log.capture()
    def _create_habitat_tab(self) -> widgets.VBox:
        """Create tab widget for habitats.

        Returns:
            widgets.VBox: widget
        """
        checkboxes = []
        pgons_checkboxes = []
        graph_checkboxes = []

        graphs = [
            (name, "graphs", layer_subtype, graph)
            for name, graph in self.layer_dict["graphs"].items()
            for layer_subtype in ["graph", "pgons"]
        ]
        maps = [
            (name, "maps", "map", map_layer["map"])
            for name, map_layer in self.layer_dict["maps"].items()
        ]
        for idx, (layer_name, layer_type, layer_subtype, layer_dict) in enumerate(
            maps + graphs
        ):

            layout = widgets.Layout(padding="0px 0px 0px 0px")

            # indenting habitat checkboxes
            if layer_type == "graphs":
                if layer_dict["is_habitat"]:
                    layout = widgets.Layout(padding="0px 0px 0px 25px")

            checkbox = widgets.Checkbox(
                value=True,
                description="{} ({})".format(layer_name, layer_subtype),
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

            if idx == 0:
                checkboxes.append(widgets.HTML("<b>Map Data</b>"))

            checkboxes.append(checkbox)

            if layer_subtype == "graph":
                graph_checkboxes.append(checkbox)
            elif layer_subtype == "pgons":
                pgons_checkboxes.append(checkbox)

            # Add habitats header if last part of main graph
            if (
                layer_type == "graphs"
                and layer_subtype == "pgons"
                and not layer_dict["is_habitat"]
            ):
                checkboxes.append(
                    widgets.HTML(
                        "<b>Habitats in {}</b>".format(layer_name),
                        layout=widgets.Layout(padding="0px 0px 0px 25px"),
                    )
                )

            # Add horizontal rule if last map to separate from graphs
            if idx == len(maps) - 1:
                checkboxes.append(widgets.HTML("<hr/>"))
                checkboxes.append(widgets.HTML("<b>Graph Data</b>"))

        # Create button to toggle all polygons at once
        hide_pgon_button = widgets.ToggleButton(description="Toggle all polygons")

        @self.log.capture()
        def hide_all_pgons(change):
            if change["name"] == "value":
                for box in pgons_checkboxes:
                    box.value = change["new"]

        hide_pgon_button.observe(hide_all_pgons)

        # Create button to toggle all graphs at once
        hide_graph_button = widgets.ToggleButton(description="Toggle all graphs")

        @self.log.capture()
        def hide_all_graphs(change):
            if change["name"] == "value":
                for box in graph_checkboxes:
                    box.value = change["new"]

        hide_graph_button.observe(hide_all_graphs)

        checkboxes.append(widgets.HTML("<hr/>"))
        buttons = widgets.HBox([hide_pgon_button, hide_graph_button])
        checkboxes.append(buttons)

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
            concise=True,
            description="Node color",
            value=self.custom_style["style"]["fillColor"],
            disabled=False,
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
        available_metrics = [
            (name, graph["metrics"])
            for name, graph in self.layer_dict["graphs"].items()
            if graph["metrics"]
        ]

        dropdown = widgets.Dropdown(
            options=[("None selected", "nothing")] + available_metrics,
            description="Graph:",
        )

        metrics_html = widgets.HTML("Select graph")

        @self.log.capture()
        def metrics_callback(change):
            metrics_str = ""
            if change["name"] == "value":
                if change["new"] != "nothing":
                    for metric in change["new"]:
                        metrics_str += """
                        <b>{}:</b> {:.2f}</br>
                        """.format(
                            metric.name, metric.value
                        )
                metrics_html.value = metrics_str

        dropdown.observe(metrics_callback)

        widget = widgets.VBox(
            [
                dropdown,
                metrics_html,
            ]
        )
        return widget

    @log.capture()
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
            Log=self.log,
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

    @log.capture()
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
