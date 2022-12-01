---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input, hide-output]

import warnings

warnings.filterwarnings('ignore')
from fmriprep_denoise.dataset.atlas import ATLAS_METADATA
from fmriprep_denoise.visualization import figures, utils

import ipywidgets as widgets
from ipywidgets import interactive, interact

path_root = utils.get_data_root() / "denoise-metrics"
```

# Results: atlas level

It is possible to view the data at atlas level!

```{code-cell} ipython3
dataset = widgets.Dropdown(
    options=['ds000228', 'ds000030'],
    description='Dataset : ',
    disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=['fmriprep-20.2.1lts', 'fmriprep-20.2.5lts'],
    description='fmriprep : ',
    disabled=False
)
atlas = widgets.Dropdown(
    options=ATLAS_METADATA.keys(),
    description='atlas : ',
    disabled=False
)
dimension = widgets.Dropdown(
        description='dimensions : ',
        disabled=False
    )

@interact(ds=dataset, f=fmriprep_version, a=atlas, d=dimension)
def print_city(ds, f, a, d):
    dimension.options = ATLAS_METADATA[a]['dimensions']
    print(ds, f, a, d)
    figures.plot_motion_resid(ds, f, path_root, atlas_name=a, dimension=d)
    figures.plot_distance_dependence(ds, f, path_root, atlas_name=a, dimension=d)
    figures.plot_network_modularity(ds, f, path_root, atlas_name=a, dimension=d)
```
