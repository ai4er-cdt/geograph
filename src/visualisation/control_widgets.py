"""Module with widgets to control GeoGraphViewer."""

from __future__ import annotations
from typing import Dict

import ipywidgets as widgets
import traitlets

from src.visualisation import geoviewer


class VisibilityWidget(widgets.Box):
    """Widget to control visibility of graphs in GeoGraphViewer."""

    # TODO: add better logging than class variable for widgets
    log_out = widgets.Output(layout={"border": "1px solid black"})

    @log_out.capture()
    def __init__(self, viewer: geoviewer.GeoGraphViewer) -> None:
        """Widget to control visibility of graphs in GeoGraphViewer.

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
                button.layer_name = self.current_map
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
