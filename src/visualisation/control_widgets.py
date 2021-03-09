"""Module with widgets to control GeoGraphViewer."""

from __future__ import annotations
from typing import Dict

import ipywidgets as widgets
import traitlets

from src.visualisation import geoviewer


class RadioVisibilityWidget(widgets.Box):
    """Widget to control visibility of graphs in GeoGraphViewer with radio buttons."""

    # TODO: add better logging than class variable for widgets
    log_out = widgets.Output(layout={"border": "1px solid black"})

    @log_out.capture()
    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to control visibility of graphs in GeoGraphViewer with radio buttons.

        This widget controls the visibility of graph as well as current map layers of
        GeoGraphViewer. Further it sets the current_graph attribute of GeoGraphViewer
        used by other widgets.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        self.viewer = viewer
        self.viewer.hidde_all_layers()
        self.graph_names = list(viewer.layer_dict["graphs"].keys())

        widget = self.assemble_widget()

        super().__init__([widget])

    @log_out.capture()
    def assemble_widget(self) -> widgets.Widget:
        """Assemble all sub-widgets making up VisibilityWidget into layout.

        Returns:
            widgets.Widget: final widget to be added to GeoGraphViewer
        """
        graph_selection = self.create_graph_selection()
        view_buttons = self.create_visibility_buttons()

        widget = widgets.VBox([graph_selection, view_buttons])

        return widget

    @log_out.capture()
    def create_graph_selection(self) -> widgets.RadioButtons:
        """Create radio buttons to enable graph selection.

        Returns:
            widgets.RadioButtons: buttons to select graph
        """
        graph_list = []
        for graph_name in self.graph_names:
            graph_str = graph_name
            graph_list.append((graph_str, graph_name))

        radio_buttons = widgets.RadioButtons(options=graph_list, description="")
        widgets.link((radio_buttons, "value"), (self.viewer, "current_graph"))

        return radio_buttons

    @log_out.capture()
    def create_visibility_buttons(self) -> widgets.Box:
        """Create buttons that toggle the visibility of current graph and map.

        The visibility of the current graph (set in self.current_graph), its subparts
        and the map (set in self.current_map) can be manipulated with the returned
        buttons. Separate buttons for the polygons and the components of the graph are
        included in the returned box.

        Returns:
            widgets.Box: box with button widgets
        """
        # Creating view buttons
        btn_layout = widgets.Layout(width="115px")
        view_graph_btn = widgets.ToggleButton(
            description="Graph",
            tooltip="View graph",
            icon="project-diagram",
            layout=btn_layout,
        )
        view_pgon_btn = widgets.ToggleButton(
            description="Polygons",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="View graph",
            icon="shapes",
            layout=btn_layout,
        )
        view_components_btn = widgets.ToggleButton(
            description="Components",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="View graph",
            icon="circle",
            layout=btn_layout,
        )
        view_map_btn = widgets.ToggleButton(
            description="Map",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="View graph",
            icon="globe-africa",
            layout=btn_layout,
        )

        button_list = [
            view_graph_btn,
            view_pgon_btn,
            view_components_btn,
            view_map_btn,
        ]

        # Setting up callback on click
        for button, layer_subtype in zip(
            button_list, ["graph", "pgons", "components", "map"]
        ):
            # Adding traits to button so we're able to access this information in
            # the callback called when clicked
            button.add_traits(
                layer_type=traitlets.Unicode().tag(sync=True),
                layer_subtype=traitlets.Unicode().tag(sync=True),
                layer_name=traitlets.Unicode().tag(sync=True),
            )
            if layer_subtype == "map":
                button.layer_type = "maps"
                button.layer_name = self.viewer.current_map
                # If current map changes the function of this button changes
                widgets.dlink((self.viewer, "current_map"), (button, "layer_name"))
            else:
                button.layer_type = "graphs"
                button.layer_name = self.viewer.current_graph
                widgets.dlink((self.viewer, "current_graph"), (button, "layer_name"))

            button.layer_subtype = layer_subtype

            button.observe(self._handle_view, names=["value", "layer_name"])

        view_graph_btn.value = True
        view_pgon_btn.value = True
        view_map_btn.value = True

        buttons = widgets.VBox(
            [widgets.HBox(button_list[:2]), widgets.HBox(button_list[2:])]
        )

        return buttons

    @log_out.capture()
    def _handle_view(self, change: Dict) -> None:
        """Callback function for trait events in view buttons"""
        owner = change.owner  # Button that is clicked or changed

        print("Detected button change:", change)

        # Accessed if button is clicked (its value changed)
        if change.name == "value":
            active = change.new
            self.viewer.set_layer_visibility(
                owner.layer_type, owner.layer_name, owner.layer_subtype, active
            )
            self.viewer.layer_update()

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
            # Note: there is a potential for speed improvement by not updating map
            # layers for each button separately, as is done here.
            self.viewer.layer_update()


class CheckboxVisibilityWidget(widgets.Box):
    """Widget to control visibility of graphs in GeoGraphViewer with checkboxes."""

    # TODO: add better logging than class variable for widgets
    log_out = widgets.Output(layout={"border": "1px solid black"})

    @log_out.capture()
    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to control visibility of graphs in GeoGraphViewer with checkboxes.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        self.viewer = viewer
        self.viewer.hidde_all_layers()
        self.graph_names = list(viewer.layer_dict["graphs"].keys())

        widget = self._create_checkboxes()

        super().__init__([widget])

    @log_out.capture()
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

        @self.log_out.capture()
        def toggle_all_pgons(change):
            if change["name"] == "value":
                for box in pgons_checkboxes:
                    box.value = change["new"]

        hide_pgon_button.observe(toggle_all_pgons)

        # Create button to toggle all graphs at once
        hide_graph_button = widgets.ToggleButton(description="Toggle all graphs")

        @self.log_out.capture()
        def toggle_all_graphs(change):
            if change["name"] == "value":
                for box in graph_checkboxes:
                    box.value = change["new"]

        hide_graph_button.observe(toggle_all_graphs)

        checkboxes.append(widgets.HTML("<hr/>"))
        buttons = widgets.HBox([hide_pgon_button, hide_graph_button])
        checkboxes.append(buttons)

        habitat_tab = widgets.VBox(checkboxes)

        return habitat_tab

    @log_out.capture()
    def _switch_layer_visibility(self, change: Dict):
        """Switch layer visibility according to change.

        Args:
            change (Dict): change dict provided by checkbox widget
        """
        if change["name"] == "value":
            owner = change["owner"]
            self.viewer.set_layer_visibility(
                owner.layer_type, owner.layer_name, owner.layer_subtype, change.new
            )

        self.viewer.layer_update()


class TimelineWidget(widgets.Box):
    """Widget to interact with GeoGraphTimeline."""

    # TODO: add better logging than class variable for widgets
    log_out = widgets.Output(layout={"border": "1px solid black"})

    @log_out.capture()
    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to interact with GeoGraphTimeline.

        Not fully implemented yet, currently just placeholder widget.

        Args:
            viewer (geoviewer.GeoGraphViewer): GeoGraphViewer to control
        """
        self.viewer = viewer
        widget = self._create_timeline_controls()

        super().__init__([widget])

    @log_out.capture()
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

        diff_tab = widgets.VBox(
            [time_slider1, time_slider2, compute_node_button, compute_pgon_button]
        )

        return diff_tab
