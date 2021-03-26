Getting Started
====================================

How to get started with the GeoGraph package.

Installation
------------

You can install geograph via pip using

.. code-block:: shell

    pip install geograph

Basic Usage
-----------

Given a variable ``data`` that is one of

- a path to a pickle file or compressed pickle file to load the graph from,
- a path to vector data in GPKG or Shapefile format,
- a path to raster data in GeoTiff format,
- a numpy array containing raster data,
- a dataframe containing polygons,

you can create a ``GeoGraph`` using

.. code-block:: python

    from geograph import GeoGraph
    graph = GeoGraph(data)


To visualise this graph use the following code in a jupyter notebook

.. code-block:: python

    from geograph.visualisation.geoviewer import GeoGraphViewer
    viewer = GeoGraphViewer()
    viewer.add_graph(graph, name='my_graph')
    viewer.enable_graph_controls()
    viewer

This should then look something like

.. image:: images/viewer_demo.gif
  :width: 500
  :alt: viewer demo

Additional Examples
-------------------

See the `jupyter notebooks in our repository`_ for more advanced examples of using GeoGraph.
You can also run them without installing GeoGraph, by using binder with `this link`_.

.. _jupyter notebooks in our repository: https://github.com/ai4er-cdt/gtc-biodiversity/tree/main/notebooks
.. _this link: https://mybinder.org/v2/gh/ai4er-cdt/gtc-biodiversity/main?urlpath=lab%2Ftree%2Fnotebooks
