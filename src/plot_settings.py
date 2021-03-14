"""
plot_settings.py
================

usage:

from src.plot_settings import ps_defaults, label_subplots, get_dim, set_dim, PALETTE

ps_defaults(use_tex=True)

# ---- example set of graphs ---

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)
x=np.linspace(0, np.pi, num=100)
axs[0, 0].plot(x, np.sin(x)); axs[0, 1].plot(x, np.cos(x))
axs[1, 0].plot(x, np.sinc(x)); axs[1, 1].plot(x, np.abs(x))

# set size
set_dim(fraction_of_line_width=1, ratio=(5**.5 - 1) / 2)

# label subplots
label_subplots(axs, start_from=0, fontsize=10)

"""
import numpy as np
import itertools
from distutils.spawn import find_executable
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import seaborn as sns


def ps_defaults(use_tex: bool=True) -> None:
    """
    Apply plotting style to produce nice looking figures.
    Call this at the start of a script which uses matplotlib,
    and choose the either high (uses latex) or low (without latex)
    """
    #'#CB4154' ,
    # matplotlib.use('agg') this used to be required for jasmin
    p_general = {
        "font.family": "serif",
        "font.serif": [],
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Set the font
        "mathtext.fontset": "cm",
        # "font.family": "STIXGeneral",
        "font.sans-serif": ["DejaVu Sans"],  # gets rid of error messages
        "font.monospace": [],
        "lines.linewidth": 0.75,
        "scatter.marker": "+",
        "image.cmap": "RdYlBu_r",
    }
    matplotlib.rcParams.update(p_general)
    matplotlib.style.use("seaborn-colorblind")

    if use_tex and find_executable("latex"):
        p_setting = {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "pgf.preamble": (r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
                             +r"\usepackage[separate -uncertainty=true]{siunitx}"),
        }
    else:
        p_setting = {
            "text.usetex": False,
        }
    matplotlib.rcParams.update(p_setting)


def label_subplots(axs: Sequence[matplotlib.axes], start_from: int=0, fontsize: int=10) -> None:
    """
    Currently adds (a), (b), (c) etc. at the top left of each subplot panel.
    Labelling order achieved through ravelling the input list/ array
    :param axs: list or array of subplot axes.
    :param start_from: lets you start from a different axes.
    :param fontsize: in points
    :return: void; alters axes objects
    """
    if isinstance(axs, list):
        axs = np.asarray(axs)
    orig_label_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
    assert len(axs.ravel()) + start_from <= len(orig_label_list)
    subset_labels = []
    for i in range(len(axs.ravel())):
        subset_labels.append(orig_label_list[i + start_from])
    for i, label in enumerate(subset_labels):
        axs.ravel()[i].text(
            0.02,
            0.95,
            str("(" + label.lower() + ")"),
            color="black",
            transform=axs.ravel()[i].transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
        )


def get_dim(fraction_of_line_width: float =1, ratio: float=(5 ** 0.5 - 1) / 2) -> Tuple[float, float]:
    """Return figure height, width in inches to avoid scaling in latex. Default is golden ratio.
    ​
        Parameters
        ----------
        fraction_of_line_width: float
                Fraction of the document width which you wish the figure to occupy

        ratio: float
                Fraction of figure width that the figure height should be
    ​
        Returns
        -------
        fig_dim: tuple
                Dimensions of figure in inches
    """
    thesis_width = 697.3  # roughly 6.5 inches

    # Width of figure
    fig_width_pt = thesis_width * fraction_of_line_width

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_dim(fig: matplotlib.pyplot.figure, fraction_of_line_width: float=1, ratio: float=(5 ** 0.5 - 1) / 2) -> None:
    """Set aesthetic figure dimensions to avoid scaling in latex.
    ​
        Parameters
        ----------
        fraction_of_line_width: float
                Fraction of the document width which you wish the figure to occupy

        ratio: float
                Fraction of figure width that the figure height should be
    ​
        Returns: void; alters current figure to have the desired dimensions
    """
    fig = plt.gcf()
    fig.set_size_inches(
        get_dim(
            fraction_of_line_width=fraction_of_line_width,
            ratio=ratio
        )
    )


_paper_colors_abbv = [
    "#4d2923ff",
    "#494f1fff",
    "#38734bff",
    "#498489ff",
    "#8481baff",
    "#c286b2ff",
    "#d7a4a3ff",
]
_paper_colors = sns.color_palette(_paper_colors_abbv)
# Note: To inspect colors, call `sns.palplot(_paper_colors)`
PALETTE = itertools.cycle(_paper_colors)
