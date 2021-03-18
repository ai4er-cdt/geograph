"""Module with utils for logging, debugging and styling ipywidgets."""

import logging
import ipywidgets as widgets
import IPython.display

# Styling widgets
HRULE = widgets.HTML('<hr style="opacity: 0.5"/>')


def create_html_header(text: str, level=1) -> widgets.HTML:
    """Create html header widget from text."""
    opacity_levels = [1.0, 0.68, 0.55, 0.4]
    opacity = opacity_levels[level]
    html_template = '<b style="opacity: {}">{}</b>'
    return widgets.HTML(
        html_template.format(opacity, text)
    )  # "<b>{}</b>".format(text))


# Logging widgets
class OutputWidgetHandler(logging.Handler):
    """Custom logging handler sending logs to an output widget.

    Copied with minor adaptations from
    https://ipywidgets.readthedocs.io/en/latest/examples/Output%20Widget.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = {
            "width": "100%",
            "border": "1px solid black",
        }
        self.out = widgets.Output(layout=layout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.setFormatter(formatter)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted_record + "\n",
        }
        self.out.outputs = self.out.outputs + (new_output,)

    def show_logs(self):
        """ Show the logs """
        IPython.display.display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()