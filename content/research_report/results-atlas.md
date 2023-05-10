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

```{code-cell}
:tags: [hide-input, hide-output]

import warnings

warnings.filterwarnings("ignore")
from fmriprep_denoise.dataset.atlas import ATLAS_METADATA
from fmriprep_denoise.visualization import figures, utils

import ipywidgets as widgets
from ipywidgets import interactive, interact

path_root = utils.get_data_root() / "denoise-metrics"
```

# Results: atlas level

It is possible to view the data at atlas level!
Please click on the launch button to lunch the Binder instance for interactive data viewing.

In the report we used four atlases, three of them came with multiple parcellation schemes.

- Gordon atlas: 333
- Schaefer 7 network atlas: 100, 200, 300, 400, 500, 600, 800
- Multiresolution Intrinsic Segmentation Template (MIST): 7, 12, 20, 36, 64, 122, 197, 325, 444, “ROI” (210 parcels, 122 split by the midline)
- DiFuMo atlas: 64, 128, 256, 512, 1024

## Before we start: Loss of temporal degrees of freedom

As any denoising strategy aims at a particular trade-off between the amount of noise removed and the preservation of degrees of freedom for signals, first and foremost, we would like to presentthe loss of temporal degrees of freedom.

This is an important consideration accompanying the remaining metrics.

```{code-cell}
:tags: [hide-input]

from fmriprep_denoise.visualization import degrees_of_freedom_loss


def notebook_plot_loss_degrees_of_freedom(criteria_name, fmriprep_version):
    datasets = ["ds000228", "ds000030"]
    data = degrees_of_freedom_loss.load_data(
        path_root, datasets, criteria_name, fmriprep_version
    )
    degrees_of_freedom_loss.plot_stats(data)


criteria_name = widgets.Dropdown(
    options=["stringent", "minimal", None],
    value="stringent",
    description="Threshould: ",
    disabled=False,
)

fmriprep_version = widgets.Dropdown(
    options=["fmriprep-20.2.1lts", "fmriprep-20.2.5lts"],
    value="fmriprep-20.2.1lts",
    description="Preporcessing version : ",
    disabled=False,
)

interactive(
    notebook_plot_loss_degrees_of_freedom,
    criteria_name=criteria_name,
    fmriprep_version=fmriprep_version,
)
```

## Each parcelation scheme

We can also plot them by each parcellation schemes.

This is the original way Ciric and colleagues (2017) presented their results!

### Gordon atlas

```{code-cell}
atlas = "gordon333"

dataset = widgets.Dropdown(
    options=["ds000228", "ds000030"], description="Dataset : ", disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=["fmriprep-20.2.1lts", "fmriprep-20.2.5lts"],
    description="fmriprep : ",
    disabled=False,
)
dimension = widgets.Dropdown(
    description="dimensions : ",
    options=ATLAS_METADATA[atlas]["dimensions"],
    disabled=False,
)


@interact(ds=dataset, f=fmriprep_version, d=dimension)
def show_atlas(ds, f, d):
    print(ds, f, ATLAS_METADATA[atlas]["atlas"], "dimensions: ", d)
    figures.plot_motion_resid(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_distance_dependence(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_network_modularity(ds, f, path_root, atlas_name=atlas, dimension=d)
```

### MIST

```{code-cell}
atlas = "mist"

dataset = widgets.Dropdown(
    options=["ds000228", "ds000030"], description="Dataset : ", disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=["fmriprep-20.2.1lts", "fmriprep-20.2.5lts"],
    description="fmriprep : ",
    disabled=False,
)
dimension = widgets.Dropdown(
    description="dimensions : ",
    options=ATLAS_METADATA[atlas]["dimensions"],
    disabled=False,
)


@interact(ds=dataset, f=fmriprep_version, d=dimension)
def show_atlas(ds, f, d):
    print(ds, f, ATLAS_METADATA[atlas]["atlas"], "dimensions: ", d)
    figures.plot_motion_resid(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_distance_dependence(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_network_modularity(ds, f, path_root, atlas_name=atlas, dimension=d)
```

### Schaefer 7 network

```{code-cell}
atlas = "schaefer7networks"

dataset = widgets.Dropdown(
    options=["ds000228", "ds000030"], description="Dataset : ", disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=["fmriprep-20.2.1lts", "fmriprep-20.2.5lts"],
    description="fmriprep : ",
    disabled=False,
)
dimension = widgets.Dropdown(
    description="dimensions : ",
    options=ATLAS_METADATA[atlas]["dimensions"],
    disabled=False,
)


@interact(ds=dataset, f=fmriprep_version, d=dimension)
def show_atlas(ds, f, d):
    print(ds, f, ATLAS_METADATA[atlas]["atlas"], "dimensions: ", d)
    figures.plot_motion_resid(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_distance_dependence(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_network_modularity(ds, f, path_root, atlas_name=atlas, dimension=d)
```

### DiFuMo

```{code-cell}
atlas = "difumo"

dataset = widgets.Dropdown(
    options=["ds000228", "ds000030"], description="Dataset : ", disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=["fmriprep-20.2.1lts", "fmriprep-20.2.5lts"],
    description="fmriprep : ",
    disabled=False,
)
dimension = widgets.Dropdown(
    description="dimensions : ",
    options=ATLAS_METADATA[atlas]["dimensions"],
    disabled=False,
)


@interact(ds=dataset, f=fmriprep_version, d=dimension)
def show_atlas(ds, f, d):
    print(ds, f, ATLAS_METADATA[atlas]["atlas"], "dimensions: ", d)
    figures.plot_motion_resid(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_distance_dependence(ds, f, path_root, atlas_name=atlas, dimension=d)
    figures.plot_network_modularity(ds, f, path_root, atlas_name=atlas, dimension=d)
```

## View as atlas collection

You can view the metrics by atlas collections here.

We will summmarise the metrics by each atlas collection.

**The summary statistics are computed on the fly, it might take a bit of time.**

```{code-cell}
dataset = widgets.Dropdown(
    options=["ds000228", "ds000030"], description="Dataset : ", disabled=False
)
fmriprep_version = widgets.Dropdown(
    options=["fmriprep-20.2.1lts", "fmriprep-20.2.5lts"],
    description="fmriprep : ",
    disabled=False,
)
atlas = widgets.Dropdown(
    description="atlas : ", options=list(ATLAS_METADATA.keys()), disabled=False
)


@interact(ds=dataset, f=fmriprep_version, a=atlas)
def show_atlas(ds, f, a):
    print(ds, f, a)
    figures.plot_motion_resid(ds, f, path_root, atlas_name=a, dimension=None)
    figures.plot_distance_dependence(ds, f, path_root, atlas_name=a, dimension=None)
    figures.plot_network_modularity(ds, f, path_root, atlas_name=a, dimension=None)
```
