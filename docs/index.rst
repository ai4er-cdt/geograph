.. GeoGraph documentation master file, created by
   sphinx-quickstart on Thu Mar 18 17:37:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home - GeoGraph Documentation
==============================
Welcome to the GeoGraph documentation!

What is GeoGraph?
-----------------

The Python package GeoGraph is built around the idea of geospatially
referenced graph - a *GeoGraph*. Given either raster
or polygon data as input, a GeoGraph is constructed by assigning each
separate patch a graph node. In a second step, edges are added between nodes whenever the
patches corresponding to two nodes are within a user-specificed distance. Based on this basic idea,
the GeoGraph package provides a wide range of visualisation and analysis tools.

What can it be used for?
------------------------

Landscape Ecology
   *Standard Analysis*
      Building on the graph-based data structure, the GeoGraph package is able to
      compute most of the standard metrics used in landscape ecology. Combined with
      an interactive user interface, it provides a powerful Python tool for
      fragmentation and connectivity analysis.

   *Policy Advice*
      Using the tools provided for landscape ecology, the GeoGraph package can be
      used to give two key insights for policy decisions:

      1. Recommend conservation areas
      2. Flag areas at potential risk of fragmentation

   *Temporal Analysis*
      The graph-based nature of the GeoGraph package allows us to track individual
      patches over time, and use this information for detailed temporal analysis of
      habitats.

Polygon Data Visualisation
   Whilst our primary use-cases are in landscape ecology, this package can be used
   to investigate any kind of polygon data files, including ``.shp`` shape files.
   The :class:`GeoGraphViewer` allows for the data can be interactively viewed.






.. toctree::
   :maxdepth: 3
   :caption: Contents:

   self
   getting_started
   tutorials
   geograph
   about


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


