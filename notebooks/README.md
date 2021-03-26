# Notebooks
## Structure
We have the following five demo notebooks you can try:
- `1-demo-landscape-metrics-comparison-to-pylandstats.ipynb`: a comparison between our method and the PyLandStats package.
- `2-demo-landscape-timeseries-metrics.ipynb`: a notebook that shows timeseries metrics.
- `3-demo-nodediff.ipynb`: a nodebook that shows node diff.
- `4-demo-geographviewer-polesia.ipynb`: a demo of our `GeoGraphViewer` user interface on the polesia data (as seen in presentation).
- `5-demo-geographviewer-chernobyl.ipynb`: a demo of our `GeoGraphViewer` user interface on ESA CCI data in the Chernobyl Exlusion Zone. This demo also shows how to use our temporal analysis of node dynamics and growth.

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