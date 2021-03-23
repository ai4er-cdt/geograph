# pylint: disable=missing-module-docstring
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "GeoGraph - Package Documentation"
copyright = (  # pylint: disable=redefined-builtin
    "2021, Herbie Bradley, Arduin Findeis, Katherine Green,"
    " Yilin Li, Simon Mathis, Simon Thomas"
)
author = ""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    'sphinx_autodoc_typehints'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "images/geograph_logo.png"

html_theme_options = {
    'style_nav_header_background': '#a0a0a0',
    'logo_only': True,
    'display_version': True,
}

# Latex options
latex_logo = "./images/geograph_logo.png"
latex_elements = {
  'extraclassoptions': 'openany,oneside',
  'papersize': 'a4paper',
  #'fncychap': '\\usepackage[Conny]{fncychap}'
}
#latex_documents = [
#  ('index', 'geograph_docs.tex', u'Package Documentation',
#   u'',
#   'manual'), #'manual'), #'howto'),
#]
