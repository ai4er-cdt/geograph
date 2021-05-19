<p align="center">
<img src="https://raw.githubusercontent.com/ai4er-cdt/geograph/main/docs/images/geograph_logo.png" alt="GeoGraph" width="300px">
</p>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ai4er-cdt/geograph/main?urlpath=lab%2Ftree%2Fnotebooks)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Documentation Status](https://readthedocs.org/projects/geograph/badge/?version=latest)](https://geograph.readthedocs.io/en/latest/?badge=latest)


__Table of contents:__
1. Description
2. Installation
3. Requirements
4. Documentation

## 1. Description

GeoGraph provides a tool for analysing habitat fragmentation and related problems in landscape ecology. GeoGraph builds a geospatially referenced graph from land cover or field survey data and enables graph-based landscape ecology analysis as well as interactive visualizations. Beyond the graph-based features, GeoGraph also enables the computation of common landscape metrics.

## 2. Installation

GeoGraph is available via pip, so you can install it using

```
pip install geograph
```

Done, you're ready to go!

You can also visit the [Github repository](https://github.com/ai4er-cdt/geograph).

See the [documentation](https://geograph.readthedocs.io/) for a full getting started guide or check out the [binder](https://mybinder.org/v2/gh/ai4er-cdt/geograph/main?urlpath=lab%2Ftree%2Fnotebooks) for tutorials on how to get started .

## 3. Requirements

GeoGraph is written in Python 3.8 and builds on [NetworkX](https://github.com/NetworkX/NetworkX), [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet), [geopandas](https://geopandas.org/), [rasterio](https://rasterio.readthedocs.io/en/latest/) and many more packages. See the [requirements directory](./requirements) for a full list of dependencies.

## 4. Documentation

Our documentation is available at [geograph.readthedocs.io](https://geograph.readthedocs.io/).
