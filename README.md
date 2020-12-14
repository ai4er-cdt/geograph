# Group Team Challenge 2021: Biodiversity Team

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

__Table of contents:__  
1. Getting started
2. Code formatting
3. Requirements
4. Project structure
## 1. Getting started
### 1.1 Cloning the project and its submodules
To get started, first clone the github repository via:  
`git clone --recurse-submodules git@github.com:ai4er-cdt/ai4er-cookiecutter.git`  
Note that this line uses the `--recurse-submodules` flag when it clones. This is needed
if you would like to also pull the overleaf report that is linked to our project repository. 
When pulling from overleaf, it will ask you for your overleaf username and password. 

If you do not want to clone the overleaf files, simply omit the `--recurse-submodules` flag.  
If you cloned the repository without the submodules, but would like to add them later, use
```
git submodule update --init
``` 

__Pulling updates from the submodule:__  
Once you have initialized the submodule, you can fetch and update any changes to the submodules via `git submodule update --remote`.  

__Pushing updates to the submodule:__  
To push updates to from your local branch to the submodule remote, simply enter the submodule directory and perform the usual `git add`, `git commit` and `git push` workflow. 

If you push an update to the superproject which includes an update to the submodule, use 
`git push --recurse-submodules=on-demand`.  

Here's the authoritative link if you'd like to learn more about [git submodules and how to work with them](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

### 1.2 Setting up the python environment
Once you cloned the repo to your system, enter the repository and run the following commands:
```
# 1. Highly recommended: 
make env  # creates a conda environment with python 3.8 and installs dependencies
conda activate ./env  # activate the environment you just created
make jupyter_pro  # activate a couple of nice jupyter notebook extensions

# 2. Optional:
# If you wish to use jupyter in dark mode
make jupyter_dark  # activates jupyter dark mode

# If you use VSCode as editor:
make vscode_pro  # Activates a couple of nice extensions in VSCode
```

## 2. Code formatting
To automatically format your code, make sure you have `black` installed (`pip install black`) and call
```black . ``` 
from within the project directory.

## 3. Requirements
- Python 3.8+

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