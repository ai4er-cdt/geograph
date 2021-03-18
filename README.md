<p align="center">
<img src="docs/images/geograph_logo.png" alt="GeoGraph" width="300px">


Created as part of the AI4ER Group Team Challenge 2021 by the Biodiversity Team.

![GeoGraphViewer demo gif](docs/images/viewer_demo.gif)
</p>

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

__Table of contents:__
1. Features
1. Getting started
1. Requirements
1. Project structure

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
GeoGrap is written in Python 3.8 and builds on [NetworkX](https://github.com/NetworkX/NetworkX), [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet) and many more packages. See the [requirements directory](./requirements) for a full list of dependencies.

## 4. Project Organization
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements`
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
|   |                     the creator's initials, and a short `-` delimited description, e.g.
|   |                     `1.0_jqp_initial-data-exploration`.
│   ├── exploratory    <- Notebooks for initial exploration.
│   └── reports        <- Polished notebooks for presentations or intermediate results.
│
├── report             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── sections       <- LaTeX sections. The report folder can be linked to your overleaf
|                         report with github submodules.
│
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data_loading   <- Scripts to download or generate data
│   │
│   ├── preprocessing  <- Scripts to turn raw data into clean data and features for modeling
|   |
│   ├── models         <- Scripts to train models and then use trained models to make
│   │                     predictions
│   │
│   └── tests          <- Scripts for unit tests of your functions
│
└── setup.cfg          <- setup configuration file for linting rules
```

---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
