"""Module with widgets to control GeoGraphViewer."""

from __future__ import annotations
from typing import Dict

import ipywidgets as widgets
import traitlets

from src.visualisation import geoviewer


class VisibilityWidget(widgets.Box):
    """Widget to control visibility of graphs in GeoGraphViewer."""

    log_out = widgets.Output(layout={"border": "1px solid black"})

    @log_out.capture()
    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        self.viewer = viewer
        self.viewer.hidde_all_layers()

        # Setting the order and current view of graph and map
        # with the latter two as traits
        self.add_traits(
            graph_index=traitlets.Int().tag(sync=True),
            map_index=traitlets.Int().tag(sync=True),
            current_graph=traitlets.Unicode().tag(sync=True),
            current_map=traitlets.Unicode().tag(sync=True),
        )

        self.graph_index = 0
        self.map_index = 0
        self.graph_names = list(viewer.layer_dict["graphs"].keys())
        self.map_names = list(viewer.layer_dict["maps"].keys())

        # If the indices change, the current_map and current_graph
        # change automatically as well.
        self._observe_index(change=None)
        self.observe(self._observe_index, names=["graph_index", "map_index"])

        # creating widget
        widget = self.assemble_widgets()

        super().__init__([widget])

    @log_out.capture()
    def assemble_widgets(self) -> widgets.Widget:
        """Assemble all sub-widgets making up VisibilityWidget into layout.

        Returns:
            widgets.Widget: final widget to be added to GeoGraphViewer
        """

        graph_selection = self.create_graph_selection()
        view_buttons = self.create_visibility_buttons()

        widget = widgets.VBox([graph_selection, view_buttons])

        return widget

    @log_out.capture()
    def _observe_index(self, change):
        """Set current graph and map based on indices"""
        print("Changed index:", change)
        self.current_graph = self.graph_names[self.graph_index]
        self.current_map = self.map_names[self.map_index]

    @log_out.capture()
    def create_graph_selection(self) -> widgets.RadioButtons:
        """Create radio buttons to enable graph selection.

        Returns:
            widgets.RadioButtons: buttons to select graph
        """

        graph_list = []
        for idx, graph_name in enumerate(self.graph_names):
            graph_str = graph_name
            graph_list.append((graph_str, idx))

        radio_buttons = widgets.RadioButtons(options=graph_list, description="")
        widgets.link((radio_buttons, "value"), (self, "graph_index"))

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

        # Adding additional traits for callback
        for button, layer_subtype in zip(
            button_list, ["graph", "pgons", "components", "map"]
        ):
            button.add_traits(
                layer_type=traitlets.Unicode().tag(sync=True),
                layer_subtype=traitlets.Unicode().tag(sync=True),
                layer_name=traitlets.Unicode().tag(sync=True),
            )
            if layer_subtype == "map":
                button.layer_type = "maps"
                button.layer_name = self.current_map
                # If current map changes the function of this button changes
                widgets.dlink((self, "current_map"), (button, "layer_name"))
            else:
                button.layer_type = "graphs"
                button.layer_name = self.current_graph
                widgets.dlink((self, "current_graph"), (button, "layer_name"))

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

        # Button clicked
        if change.name == "value":
            active = change.new
            self.viewer.set_layer_visibility(
                owner.layer_type, owner.layer_name, owner.layer_subtype, active
            )
            self.viewer.layer_update()

        # Layer that the button is assigned is changed
        elif change.name == "layer_name":
            new_layer_name = change.new
            old_layer_name = change.old

            # remove old layer
            self.viewer.set_layer_visibility(
                owner.layer_type, old_layer_name, owner.layer_subtype, False
            )
            # view new layer
            self.viewer.set_layer_visibility(
                owner.layer_type, new_layer_name, owner.layer_subtype, owner.value
            )
            # Note: there is a potential for speed improvement by not updating map
            # layers for each button separately, as is done here.
            self.viewer.layer_update()
