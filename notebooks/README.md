# Notebooks
## Structure
We have the following five demo notebooks you can try:
- `1-demo-landscape-metrics-comparison-to-pylandstats.ipynb`: a comparison between our method and the PyLandStats package.
- `2-demo-landscape-timeseries-metrics.ipynb`: an illustration on how to use GeoGraph for time-series analysis of landscape-level, class-level, habitat-level and patch-distribution-level metrics around the Chernobyl exclusion zone.
- `3-demo-nodediff.ipynb`: an investigation of the spatially resolved qualitative and quantitative dynamics of land cover in the Chernobyl Exclusion Zone.
- `4-demo-geographviewer-polesia.ipynb`: a demo of our `GeoGraphViewer` user interface on the Polesia data<sup>1</sup> (as seen in presentation).
- `5-demo-geographviewer-chernobyl.ipynb`: a demo of our `GeoGraphViewer` user interface on ESA CCI data in the Chernobyl Exclusion Zone. This demo also shows how to use our temporal analysis of node dynamics and growth.

<sup>1</sup>The Polesia data set used here was created by Dmitri Grummo, for the Endangered Landscapes Program (https://www.endangeredlandscapes.org/), and funded by Arcadia, a charitable fund of Lisbet Rausing and Peter Baldwin.

## Useful initialization cell
To avoid having to reload the notebook when you change code from underlying imports, we recommend the following handy initialization cell for jupyter notebooks:
```
%load_ext autoreload             # loads the autoreload package into ipython kernel
%autoreload 2                    # sets autoreload mode to automatically reload modules when they change
%config IPCompleter.greedy=True  # enables tab completion
```


```
from jupyterthemes import jtplot
jtplot.style(theme=’monokai’, context=’notebook’, ticks=True, grid=False)
```
