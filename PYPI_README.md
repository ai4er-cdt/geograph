<p align="center">
<img src="https://raw.githubusercontent.com/ai4er-cdt/gtc-biodiversity/main/docs/images/geograph_logo.png" alt="GeoGraph" width="300px">
</p>

_Created as part of the AI4ER Group Team Challenge 2021 by the Biodiversity Team._


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ai4er-cdt/gtc-biodiversity/main?urlpath=lab%2Ftree%2Fnotebooks)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Documentation Status](https://readthedocs.org/projects/geograph/badge/?version=latest)](https://geograph.readthedocs.io/en/latest/?badge=latest)


__Table of contents:__
1. Features
2. Getting started
3. Requirements
4. Documentation
5. Project structure

## 1. Features

GeoGraph provides a full-stack tool for analysing habitat fragmentation, and other related problems. It includes models to predict land cover classes, a method to extract graph structure from the resulting land cover maps and an extensive range of visualisation and analysis tools.

## 2. Getting started

Clone this repository using

```
git clone https://github.com/ai4er-cdt/gtc-biodiversity.git
```

Enter the directory and install the conda environment using

```
cd gtc-biodiversity
make env
```

Done, you're ready to go!

## 3. Requirements

GeoGraph is written in Python 3.8 and builds on [NetworkX](https://github.com/NetworkX/NetworkX), [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet) and many more packages. See the [requirements directory](./requirements) for a full list of dependencies.

## 4. Documentation

Our documentation is available at [geograph.readthedocs.io](https://geograph.readthedocs.io/).

