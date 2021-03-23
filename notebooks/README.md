# Notebooks
## Structure
TODO

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