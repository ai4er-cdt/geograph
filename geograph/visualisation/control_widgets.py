"""Module with widgets to control GeoGraphViewer."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import ipywidgets as widgets
import traitlets
from geograph.visualisation import geoviewer, widget_utils


class BaseControlWidget(widgets.Box):
    """Base class for control widgets."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Base class for control widgets.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        super().__init__()
        self.viewer = viewer

        # Setting log with handler, that allows access to log
        # via self.log_handler.show_logs()
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(self.viewer.logger.level)
        self.log_handler = self.viewer.log_handler
        self.logger.addHandler(self.log_handler)

        self.logger.info("BaseControlWidget initialised.")


class GraphControlWidget(BaseControlWidget):
    """Widget with full set of controls for GeoGraphViewer."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget with full set of controls for GeoGraphViewer.

        This is the control widget added to GeoGraphViewer. It is directly added to the
        viewer and combines other widgets such as visbility control, metrics, settings
        and more.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        super().__init__(viewer=viewer)

        # Creating individual (sub-)widgets
        visibility_widget = RadioVisibilityWidget(viewer=self.viewer)
        metrics_widget = MetricsWidget(viewer=self.viewer)
        settings_widget = SettingsWidget(viewer=self.viewer)

        viewer_height = int(viewer.layout.height.replace("px", ""))
        metrics_widget.layout.height = "{}px".format(viewer_height * 0.3)

        if self.viewer.small_screen:
            view_tab = [visibility_widget]
        else:
            view_tab = [visibility_widget, widget_utils.HRULE, metrics_widget]

        # Create combined widget, each key corresponds to a tab
        combined_widget_dict = dict()
        combined_widget_dict["View"] = widgets.VBox(view_tab)
        if self.viewer.small_screen:
            combined_widget_dict["Metrics"] = metrics_widget
        combined_widget_dict["Settings"] = settings_widget
        combined_widget_dict["Log"] = self.log_handler.out

        combined_widget = widgets.Tab()
        combined_widget.children = list(combined_widget_dict.values())
        for i, title in enumerate(combined_widget_dict):
            combined_widget.set_title(i, title)

        self.children = [combined_widget]


class RadioVisibilityWidget(BaseControlWidget):
    """Widget to control visibility of graphs in GeoGraphViewer with radio buttons."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to control visibility of graphs in GeoGraphViewer with radio buttons.

        This widget controls the visibility of graph as well as current map layers of
        GeoGraphViewer. Further, it sets the current_graph attribute of GeoGraphViewer
        that controls its state and is used by other widgets.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        super().__init__(viewer=viewer)

        # Resetting all prior visibility control
        self.viewer.hide_all_layers()

        widget = self.assemble_widget()
        self.children = [widget]

    def assemble_widget(self) -> widgets.Widget:
        """Assemble all sub-widgets making up VisibilityWidget into layout.

        Returns:
            widgets.Widget: final widget to be added to GeoGraphViewer
        """
        graph_selection = self._create_layer_selection(layer_type="graphs")
        map_selection = self._create_layer_selection(layer_type="maps")
        view_buttons = self.create_visibility_buttons()

        widget = widgets.VBox(
            [
                widget_utils.create_html_header("Graph Selection"),
                graph_selection,
                widget_utils.HRULE,
                widget_utils.create_html_header("Map Selection"),
                map_selection,
                widget_utils.HRULE,
                widget_utils.create_html_header("View Selection"),
                view_buttons,
            ]
        )

        return widget

    def _create_layer_selection(
        self, layer_type: str = "graphs"
    ) -> widgets.RadioButtons:
        """Create radio buttons to enable layer selection.

        Args:
            layer_type (str, optional): one of "graphs" or "maps". Defaults to "graphs".

        Returns:
            widgets.RadioButtons: buttons to select from available layers of layer_type
        """
        layer_list = []
        layers = list(self.viewer.layer_dict[layer_type].items())
        for layer_name, layer in layers:
            layer_str = layer_name
            if layer_type == "graphs" and layer["is_habitat"]:
                layer_str += " (habitat of {})".format(layer["parent"])
            layer_list.append((layer_str, layer_name))

        radio_buttons = widgets.RadioButtons(
            options=layer_list, description="", layout={"width": "max-content"}
        )
        if layer_type == "graphs":
            viewer_attr = "current_graph"
        elif layer_type == "maps":
            viewer_attr = "current_map"
        widgets.link((radio_buttons, "value"), (self.viewer, viewer_attr))

        return radio_buttons

    def create_visibility_buttons(self) -> widgets.Box:
        """Create buttons that toggle the visibility of current graph and map.

        The visibility of the current graph (set in self.current_graph), its subparts
        (e.g. components, disconnected nodes, etc.) and the map (set in
        self.current_map) can be controlled with the returned buttons. Separate
        buttons for the polygons and the components of the graph are included in the
        returned box.

        Returns:
            widgets.Box: box with button widgets
        """

        view_graph_btn = LayerButtonWidget(
            description="Graph",
            tooltip="View graph",
            icon="project-diagram",
            layer_type="graphs",
            layer_subtype="graph",
            viewer=self.viewer,
        )
        view_pgon_btn = LayerButtonWidget(
            description="Polygons",
            tooltip="View polygons",
            icon="shapes",
            layer_type="graphs",
            layer_subtype="pgons",
            viewer=self.viewer,
        )
        view_components_btn = LayerButtonWidget(
            description="Components",
            tooltip=(
                "View components of graph. If current graph is habitat, components show"
                " the reach of an animal in a component (based on max_travel_distance)."
            ),
            icon="circle",
            layer_type="graphs",
            layer_subtype="components",
            viewer=self.viewer,
        )
        view_map_btn = LayerButtonWidget(
            description="Map",
            tooltip="View map",
            icon="globe-africa",
            layer_type="maps",
            layer_subtype="map",
            viewer=self.viewer,
        )
        view_disconnected_nodes_btn = LayerButtonWidget(
            description="Disconnected patches",
            tooltip="View disconnected patches (patches with no edge)",
            icon="exclamation-circle",
            layer_type="graphs",
            layer_subtype="disconnected_nodes",
            viewer=self.viewer,
        )
        view_poorly_con_nodes_btn = LayerButtonWidget(
            description="Poorly conn. patches",
            tooltip="View poorly connected patches (patches with single edge)",
            icon="exclamation-circle",
            layer_type="graphs",
            layer_subtype="poorly_connected_nodes",
            viewer=self.viewer,
        )
        node_dynamics_btn = LayerButtonWidget(
            description="Show node dynamics",
            tooltip="Show node dynamics.",
            icon="exclamation-circle",
            layer_type="graphs",
            layer_subtype="node_dynamics",
            viewer=self.viewer,
        )
        node_change_btn = LayerButtonWidget(
            description="Show node growth",
            tooltip="View node absolute growth. See hover widget for patch values.",
            icon="exclamation-circle",
            layer_type="graphs",
            layer_subtype="node_change",
            viewer=self.viewer,
        )

        view_graph_btn.value = True
        view_pgon_btn.value = True
        view_map_btn.value = True

        buttons = widgets.TwoByTwoLayout(
            top_left=view_graph_btn,
            top_right=view_pgon_btn,
            bottom_left=view_components_btn,
            bottom_right=view_map_btn,
        )

        buttons = widgets.VBox(
            [
                widget_utils.create_html_header("Main", level=2),
                widgets.HBox([view_graph_btn, view_pgon_btn, view_map_btn]),
                widget_utils.create_html_header("Insights", level=2),
                widgets.VBox(
                    [
                        view_components_btn,
                        view_disconnected_nodes_btn,
                        view_poorly_con_nodes_btn,
                        node_dynamics_btn,
                        node_change_btn,
                    ]
                ),
            ]
        )

        return buttons


class LayerButtonWidget(widgets.ToggleButton):
    """Toggle button to change the visibility of GeoGraphViewer layer."""

    def __init__(
        self,
        viewer: geoviewer.GeoGraphViewer,
        layer_type: str,
        layer_subtype: str,
        layer_name: Optional[str] = None,
        link_to_current_state: bool = True,
        layout: Optional[widgets.Layout] = None,
        **kwargs,
    ) -> None:
        """Toggle button to change the visibility of GeoGraphViewer layer.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
            layer_type (str): type of layer
            layer_subtype (str): subtype of layer
            layer_name (Optional[str], optional): name of layer. Defaults to None. If
                None, the layer_name is automatically set to viewer.current_graph or
                viewer.current_map (depending on layer_type).
            link_to_current_state (bool, optional): whether a traitlets link between
                the current state of the viewer and the button layer_name should be
                created. Defaults to True.
            layout (Optional[widgets.Layout], optional): layout of the button.
                Defaults to None.
        """

        self.viewer = viewer

        # Setting log with handler, that allows access to log
        # via self.log_handler.show_logs()
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(self.viewer.logger.level)
        self.log_handler = self.viewer.log_handler
        self.logger.addHandler(self.log_handler)

        if layout is None:
            layout = widgets.Layout(height="auto", width="auto")

        super().__init__(layout=layout, **kwargs)

        self.add_traits(
            layer_type=traitlets.Unicode().tag(sync=True),
            layer_subtype=traitlets.Unicode().tag(sync=True),
            layer_name=traitlets.Unicode().tag(sync=True),
        )
        self.layer_subtype = layer_subtype
        self.layer_type = layer_type

        if layer_type == "maps":
            if layer_name is None:
                layer_name = self.viewer.current_map
            self.layer_name = layer_name

            # If current map changes the function of this button changes
            if link_to_current_state:
                widgets.dlink((self.viewer, "current_map"), (self, "layer_name"))

        elif layer_type == "graphs":
            if layer_name is None:
                layer_name = self.viewer.current_graph
            self.layer_name = layer_name

            if link_to_current_state:
                widgets.dlink((self.viewer, "current_graph"), (self, "layer_name"))

        self.observe(self._handle_view, names=["value", "layer_name"])
        self._check_layer_exists()

        self.logger.info("Initialised.")

    def _handle_view(self, change: Dict) -> None:
        """Callback function for trait events in view buttons"""
        try:
            self.logger.info(
                "LayerButtonWidget callback started for %s of %s. (type: %s)",
                self.layer_subtype,
                self.layer_name,
                self.layer_type,
            )

            owner = change.owner  # Button that is clicked or changed

            # Accessed if button is clicked (its value changed)
            if change.name == "value":
                active = change.new
                self.viewer.set_layer_visibility(
                    owner.layer_type, owner.layer_name, owner.layer_subtype, active
                )
                self.viewer.request_layer_update()

            # Accessed if layer that the button is assigned to was changed
            elif change.name == "layer_name":
                new_layer_name = change.new
                old_layer_name = change.old

                # make old layer invisible
                self.viewer.set_layer_visibility(
                    owner.layer_type, old_layer_name, owner.layer_subtype, False
                )
                # make new layer visible
                self.viewer.set_layer_visibility(
                    owner.layer_type, new_layer_name, owner.layer_subtype, owner.value
                )

                self._check_layer_exists()
                # Note: there is a potential for speed improvement by not updating map
                # layers for each button separately, as is done here.
                self.viewer.request_layer_update()
        except:  # pylint: disable=bare-except
            self.logger.exception(
                "Exception in LayerButtonWidget callback on button click or change."
            )

    def _check_layer_exists(self) -> None:
        """Check if layer exists and hide button if it doesn't."""
        layer_exists = (
            self.viewer.layer_dict[self.layer_type][self.layer_name][
                self.layer_subtype
            ]["layer"]
            is not None
        )
        # hide button if layer doesn't exist
        if layer_exists:
            self.layout.display = "block"
        else:
            self.layout.display = "none"
            self.logger.debug(
                (
                    "LayerButtonWidget hidden for %s of %s. "
                    "(type: %s). Layer doesn't exist."
                ),
                self.layer_subtype,
                self.layer_name,
                self.layer_type,
            )


class CheckboxVisibilityWidget(BaseControlWidget):
    """Widget to control visibility of graphs in GeoGraphViewer with checkboxes."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to control visibility of graphs in GeoGraphViewer with checkboxes.

        Note: this is currently not used by the main GraphControlWidget.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        super().__init__(viewer=viewer)

        widget = self._create_checkboxes()
        self.children = [widget]

    def _create_checkboxes(self) -> widgets.VBox:
        """Create widget with checkbox for each layer.

        Returns:
            widgets.VBox: widget
        """
        checkboxes = []
        pgons_checkboxes = []
        graph_checkboxes = []

        graphs = [
            (name, "graphs", layer_subtype, graph)
            for name, graph in self.viewer.layer_dict["graphs"].items()
            for layer_subtype in ["graph", "pgons"]
        ]
        maps = [
            (name, "maps", "map", map_layer["map"])
            for name, map_layer in self.viewer.layer_dict["maps"].items()
        ]

        # Add checkboxes for all maps and graphs (including habitats)
        for idx, (layer_name, layer_type, layer_subtype, layer_dict) in enumerate(
            maps + graphs
        ):

            layout = widgets.Layout(padding="0px 0px 0px 0px")

            # Indent habitat checkboxes
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

        def toggle_all_pgons(change):
            try:
                if change["name"] == "value":
                    for box in pgons_checkboxes:
                        box.value = change["new"]
            except:  # pylint: disable=bare-except
                self.logger.exception("Exception in view button callback on click.")

        hide_pgon_button.observe(toggle_all_pgons)

        # Create button to toggle all graphs at once
        hide_graph_button = widgets.ToggleButton(description="Toggle all graphs")

        def toggle_all_graphs(change):
            try:
                if change["name"] == "value":
                    for box in graph_checkboxes:
                        box.value = change["new"]
            except:  # pylint: disable=bare-except
                self.logger.exception("Exception in view button callback on click.")

        hide_graph_button.observe(toggle_all_graphs)

        checkboxes.append(widgets.HTML("<hr/>"))
        buttons = widgets.HBox([hide_pgon_button, hide_graph_button])
        checkboxes.append(buttons)

        return widgets.VBox(checkboxes)

    def _switch_layer_visibility(self, change: Dict):
        """Switch layer visibility according to change.

        Args:
            change (Dict): change dict provided by checkbox widget
        """
        try:
            self.logger.debug("Checkbox callback called.")
            if change["name"] == "value":
                owner = change["owner"]
                self.viewer.set_layer_visibility(
                    owner.layer_type, owner.layer_name, owner.layer_subtype, change.new
                )

                self.viewer.layer_update()
        except:  # pylint: disable=bare-except
            self.logger.exception("Exception in view checkbox callback on click.")


class TimelineWidget(BaseControlWidget):
    """Widget to interact with GeoGraphTimeline."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to interact with GeoGraphTimeline.

        Note: not fully implemented yet, currently just placeholder widget.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        super().__init__(viewer=viewer)

        widget = self._create_timeline_controls()
        self.children = [widget]

    def _create_timeline_controls(self) -> widgets.VBox:
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
            button_style="",
            tooltip=(
                "This computes the differences between of the nodes in the graph"
                " at start time and the graph at end time."
            ),
            icon="",
        )

        compute_pgon_button = widgets.Button(
            description="Compute polygon diff",
            disabled=False,
            button_style="",
            tooltip=(
                "This computes the differences between of the polygons in the"
                " graph at start time and the graph at end time."
            ),
            icon="",
        )

        timeline_widget = widgets.VBox(
            [time_slider1, time_slider2, compute_node_button, compute_pgon_button]
        )

        return timeline_widget


class MetricsWidget(BaseControlWidget):
    """Widget to show graph metrics in GeoGraphViewer."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to show graph metrics in GeoGraphViewer.

        This widget shows metrics for `viewer.current_graph`.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to show metrics for
        """
        super().__init__(viewer=viewer)

        metrics_widget = self._create_metrics_widget()
        self.children = [
            widgets.VBox(
                [widget_utils.create_html_header("Graph Metrics"), metrics_widget]
            )
        ]

    def _create_metrics_widget(self) -> widgets.VBox:
        """Create metrics visualisation widget.

        Returns:
            widgets.VBox: metrics widget
        """

        metrics_html = widgets.HTML("No graph selected.")

        # Added to ensure vertical scroll bar on right hand-side of box
        # TODO: find a better fix to ensure width of all sub-widgets are the same
        metrics_html.layout.width = "242px"
        metrics_html.layout.overflow_x = "hidden"

        def metrics_callback(change):
            try:
                self.logger.debug("MetricsWidget callback called.")

                graph_name = change["new"]
                graph_layer = change["owner"].layer_dict["graphs"][graph_name]
                graph = graph_layer["original_graph"]
                metrics = graph_layer["metrics"]

                # Adding general and habitat graph information
                metrics_str = widget_utils.create_html_header(
                    "Information", level=2
                ).value

                if graph_layer["is_habitat"]:
                    information = {
                        "parent": graph_layer["parent"],
                        "valid_classes": graph.valid_classes,
                        "max_travel_distance": graph.max_travel_distance,
                        # "barrier_classes": graph_layer["barrier_classes"] #TODO: add
                    }
                else:
                    information = {}
                information["Number of edges"] = len(graph.graph.edges())
                information["Number of nodes"] = len(graph.graph.nodes())

                for info_name, info_value in information.items():
                    metrics_str += "</br><b>{}:</b> {}".format(info_name, info_value)
                metrics_str += "</br>"

                # Adding metrics
                # TODO: avoid creating a widget just for this string
                # may lead to inefficiency
                metrics_str += widget_utils.create_html_header("Metrics", level=2).value
                if metrics:
                    for metric in metrics:
                        metrics_str += "</br><b>{}:</b> {:.2f}".format(
                            metric.name, metric.value
                        )
                else:
                    metrics_str += " No metrics available for current graph."
                metrics_html.value = metrics_str
            except:  # pylint: disable=bare-except
                self.logger.exception(
                    "Exception in metrics callback after current_graph change."
                )

        self.viewer.observe(metrics_callback, names=["current_graph"])
        metrics_callback(dict(new=self.viewer.current_graph, owner=self.viewer))

        return metrics_html


class SettingsWidget(BaseControlWidget):
    """Widget for settings in GeoGraphViewer."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget for settings in GeoGraphViewer.

        Enables setting node size and color, and zoom level of viewer.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to show settings for
        """
        super().__init__(viewer=viewer)

        widget = self._create_settings_widget()
        self.children = [widget]

    def _create_settings_widget(self) -> widgets.VBox:
        """Create settings widget.

        Returns:
            widgets.VBox: settings widget
        """

        radius_slider = widgets.FloatSlider(
            min=0.01, max=100.0, step=0.005, value=5.0, description="Node radius:"
        )

        node_color_picker = widgets.ColorPicker(
            concise=True,
            description="Node color",
            value=self.viewer.layer_style["graph"]["style"]["fillColor"],
            disabled=False,
        )

        self._widget_output = widgets.interactive_output(
            self._set_graph_style_callback,
            dict(radius=radius_slider, node_color=node_color_picker),
        )

        zoom_slider = widgets.FloatSlider(
            description="Zoom level:", min=0, max=15, value=7
        )
        widgets.jslink((zoom_slider, "value"), (self.viewer, "zoom"))

        settings_widget = widgets.VBox(
            [
                zoom_slider,
                node_color_picker,
                radius_slider,
            ]
        )

        return settings_widget

    def _set_graph_style_callback(self, *args, **kwargs):
        """Callback function to set graph style in geoviewer.

        Args:
            args, kwargs: arguments to be passed on to viewer.set_graph_style().
        """
        try:
            self.logger.debug("Callback: graph style callback started.")
            self.viewer.set_graph_style(*args, **kwargs)
        except:  # pylint: disable=bare-except
            self.logger.exception("Exception in when setting graph style.")


class HoverWidget(BaseControlWidget):
    """Widget for showing patch information on mouse hover in GeoGraphViewer."""

    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget for showing patch information on mouse hover in GeoGraphViewer.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to show patch info in.
        """
        super().__init__(viewer=viewer)

        widget = self._create_hover_widget()
        self.children = [widget]

    def _create_hover_widget(self) -> widgets.VBox:
        """Create hover widget.

        Returns:
            widgets.VBox: settings widget
        """
        self.hover_html = widgets.HTML("""Hover over patches""")
        self.hover_html.layout.margin = "10px 10px 10px 10px"
        self.hover_html.layout.max_width = "300px"

        # Add callback to all patch (pgon) layer of graphs
        for graph_dict in self.viewer.layer_dict["graphs"].values():
            pgon_choropleth = graph_dict["pgons"]["layer"]
            pgon_choropleth.on_hover(self._hover_callback)

            # Enable hover for node dynamics
            node_dynamics_choropleth = graph_dict["node_dynamics"]["layer"]
            if node_dynamics_choropleth is not None:
                node_dynamics_choropleth.on_hover(self._hover_callback)

            # Enable hover for node absolute growth (change)
            abs_growth_choropleth = graph_dict["node_change"]["layer"]
            if abs_growth_choropleth is not None:
                abs_growth_choropleth.on_hover(self._hover_callback)

        return widgets.VBox([self.hover_html])

    def _hover_callback(self, feature, **kwargs):  # pylint: disable=unused-argument
        """Callback function on hover on graph polygon patch"""
        try:
            # self.logger.debug("HoverWidget callback called.")
            new_value = widget_utils.create_html_header(
                "Current Patch"
            ).value + """</br>
                <b>Class label:</b> {}</br>
                <b>Area:</b> {:.2f} ha</br>
                <b>Perimeter:</b> {:.2f} km</br>
                <b>Shape index:</b> {:.2f}</br>
                <b>Fractal dim.:</b> {:.2f}
            """.format(
                feature["properties"]["class_label"],
                feature["properties"]["area"] / 1e4,
                feature["properties"]["perimeter"] / 1e3,
                feature["properties"]["shape_index"],
                feature["properties"]["fractal_dimension"],
            )
            if "node_dynamic" in feature["properties"].keys():
                new_value += "</br><b>Node dyanmic:</b> {}".format(
                    feature["properties"]["node_dynamic"]
                )
            if "absolute_growth" in feature["properties"].keys():
                new_value += "</br><b>Abs. growth:</b> {:.2f} ha/yr".format(
                    feature["properties"]["absolute_growth"] / 1e4
                )
            self.hover_html.value = new_value
        except:  # pylint: disable=bare-except
            self.logger.exception("Exception in hover callback.")
