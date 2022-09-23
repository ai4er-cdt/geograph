"""
plot_settings.py
================

usage:

from geograph.plot_settings import (
    ps_defaults,
    label_subplots,
    get_dim,
    set_dim,
    PALETTE,
    STD_CLR_LIST,
    CAM_BLUE,
    BRICK_RED,
    OX_BLUE,
)

ps_defaults(use_tex=True)

# ---- example set of graphs ---

import numpy as np
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 2)

x = np.linspace(0, np.pi, num=100)
axs[0, 0].plot(x, np.sin(x), color=STD_CLR_LIST[0])
axs[0, 1].plot(x, np.cos(x), color=STD_CLR_LIST[1])
axs[1, 0].plot(x, np.sinc(x), color=STD_CLR_LIST[2])
axs[1, 1].plot(x, np.abs(x), color=STD_CLR_LIST[3])

# set size
set_dim(fig, fraction_of_line_width=1, ratio=(5 ** 0.5 - 1) / 2)

# label subplots
label_subplots(axs, start_from=0, fontsize=10)

"""
import itertools
from distutils.spawn import find_executable
from typing import Sequence, Tuple

import matplotlib
import matplotlib.style
import numpy as np
import seaborn as sns


def ps_defaults(use_tex: bool = True) -> None:
    """Apply plotting style to produce nice looking figures.
    Call this at the start of a script which uses `matplotlib`.
    Can enable `matplotlib` LaTeX backend if it is available.

    Args:
        use_tex (bool, optional): Whether or not to use latex matplotlib backend.
            Defaults to True.

    Example::
        >>> from geograph.demo.plot_settings import ps_defaults
        >>> ps_defaults(use_tex=False)
    """
    # matplotlib.use('agg') this used to be required for jasmin
    p_general = {
        "font.family": "STIXGeneral",  # Nice alternative font.
        # "font.family": "serif",
        # "font.serif": [],
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Set the font for maths
        "mathtext.fontset": "cm",
        # "font.sans-serif": ["DejaVu Sans"],  # gets rid of error messages
        # "font.monospace": [],
        "lines.linewidth": 1.0,
        "scatter.marker": "+",
        "image.cmap": "RdYlBu_r",
    }
    matplotlib.rcParams.update(p_general)
    matplotlib.style.use("seaborn-colorblind")

    if use_tex and find_executable("latex"):
        p_setting = {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "pgf.preamble": (
                r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
                + r"\usepackage[separate -uncertainty=true]{siunitx}"
            ),
        }
    else:
        p_setting = {
            "text.usetex": False,
        }
    matplotlib.rcParams.update(p_setting)


def label_subplots(
    axs: Sequence[matplotlib.pyplot.axes],
    labels: Sequence[str] = [chr(ord("`") + z) for z in range(1, 27)],
    start_from: int = 0,
    fontsize: int = 10,
    x_pos: float = 0.02,
    y_pos: float = 0.95,
) -> None:
    """Adds (a), (b), (c) at the top left of each subplot panel.
    Labelling order achieved through ravelling the input list / array.

    Args:
        axs (Sequence[matplotlib.axes]): list or array of subplot axes.
        labels (Sequence[str]): A sequence of labels for the subplots.
        start_from (int, optional): skips first ${start_from} labels. Defaults to 0.
        fontsize (int, optional): Font size for labels. Defaults to 10.
        x_pos (float, optional): Relative x position of labels. Defaults to 0.02.
        y_pos (float, optional): Relative y position of labels. Defaults to 0.95.

    Returns:
        void; alters the `matplotlib.pyplot.axes` objects

    """
    if isinstance(axs, list):
        axs = np.asarray(axs)
    assert len(axs.ravel()) + start_from <= len(labels)
    subset_labels = []
    for i in range(len(axs.ravel())):
        subset_labels.append(labels[i + start_from])
    for i, label in enumerate(subset_labels):
        axs.ravel()[i].text(
            x_pos,
            y_pos,
            str("(" + label + ")"),
            color="black",
            transform=axs.ravel()[i].transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
        )


def get_dim(
    width: float = 600,
    fraction_of_line_width: float = 1,
    ratio: float = (5**0.5 - 1) / 2,
) -> Tuple[float, float]:
    """Return figure height, width in inches to avoid scaling in latex.

       Default is golden ratio, with figur occupying full page width.

    Args:
        width (float): Textwidth of the report to make fontsizes match.
        fraction_of_line_width (float, optional): Fraction of the document width
            which you wish the figure to occupy.  Defaults to 1.
        ratio (float, optional): Fraction of figure width that the figure height
            should be. Defaults to (5 ** 0.5 - 1)/2.

    Returns:
        fig_dim (tuple):
            Dimensions of figure in inches
    """

    # Width of figure
    fig_width_pt = width * fraction_of_line_width

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_dim(
    fig: matplotlib.pyplot.figure,
    width: float = 600,
    fraction_of_line_width: float = 1,
    ratio: float = (5**0.5 - 1) / 2,
) -> None:
    """Set aesthetic figure dimensions to avoid scaling in latex.

    Args:
        fig (matplotlib.pyplot.figure): Figure object to resize.
        width (float): Textwidth of the report to make fontsizes match.
        fraction_of_line_width (float, optional): Fraction of the document width
            which you wish the figure to occupy.  Defaults to 1.
        ratio (float, optional): Fraction of figure width that the figure height
            should be. Defaults to (5 ** 0.5 - 1)/2.

    Returns:
        void; alters current figure to have the desired dimensions
    """
    fig.set_size_inches(
        get_dim(width=width, fraction_of_line_width=fraction_of_line_width, ratio=ratio)
    )


STD_CLR_LIST = [
    "#4d2923ff",
    "#494f1fff",
    "#38734bff",
    "#498489ff",
    "#8481baff",
    "#c286b2ff",
    "#d7a4a3ff",
]
_paper_colors = sns.color_palette(STD_CLR_LIST)
# Note: To inspect colors, call `sns.palplot(_paper_colors)`
PALETTE = itertools.cycle(_paper_colors)
CAM_BLUE = "#a3c1ad"
OX_BLUE = "#002147"
BRICK_RED = "#CB4154"
