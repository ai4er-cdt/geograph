#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that defines a setup function and publishes the package to PyPI.

Use the command `python setup.py upload`.
"""
# Note: To use the "upload" functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree
from typing import Dict, List

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "geograph"
DESCRIPTION = "Group Team Challenge 2021 - Biodiversity Team"
URL = "https://geograph.readthedocs.io/"
EMAIL = "hb574@cam.ac.uk"
AUTHOR = "Biodiversity Team"
REQUIRES_PYTHON = ">=3.8.0"

# What packages are required for this module to be executed?
REQUIRED: List = [
    "numpy",
    "pandas",
    "folium",
    "ipyleaflet",
    "tqdm",
    "geopandas",
    "shapely",
    "rtree",
    "rasterio",
    "xarray",
    "networkx",
]

# What packages are optional?
EXTRAS: Dict = {
    # "fancy feature": ["django"],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the PYPI README and use it as the long-description.
# Note: this will only work if "PYPI_README.md" is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "PYPI_README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package"s _version.py module as a dictionary.
about: Dict = {}
with open(os.path.join(here, NAME, "_version.py")) as f:
    # pylint: disable=exec-used
    exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options: List = []

    @staticmethod
    def status(s):
        """Print things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Publish package to PyPI."""
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # entry_points={},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: GIS",
        "Typing :: Typed",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
    test_suite="geograph.tests.test_all.suite",
)
