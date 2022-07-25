---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Results

Overall, the current benchmark show consistent results in the two datasets.
The trend of benchmaker metrics does not differ amongst the choice of atlases.
However, we can see variance within the parcellation scheme MIST and DiFuMo.
The variance comes from different resolution of the same parcellation scheme, epecially with low parcel counts.

```{code-cell} ipython3
:tags: [hide-input, remove-output]
import warnings
warnings.filterwarnings("ignore")
from fmriprep_denoise.visualization import figures

from myst_nb import glue


plot_info = 'dof'

fig = figures.plot_dof_overview(plot_info)
glue("dof-fig", fig, display=False)

datasets = ["ds000228", "ds000030"]
for dataset in datasets:
    fig = figures.plot_dof_dataset(dataset, plot_info)
    glue(f"dof-{dataset}-fig", fig, display=False)

plot_info = 'scrubbing'

fig = figures.plot_dof_overview(plot_info)
glue("scrubbing-fig", fig, display=False)

datasets = ["ds000228", "ds000030"]
for dataset in datasets:
    fig = figures.plot_dof_dataset(dataset, plot_info)
    glue(f"scrubbing-{dataset}-fig", fig, display=False)

```

## The loss in temporal degrees of freedom in different strategies

When selecting denoising strategy, the loss in degrees of freedom `simple` and `simple+gsr` used the least amount of regressors. 
In {cite:t}`ciric_benchmarking_2017`, the equivalent `aroma` and `aroma+gsr` strategies were reported with a lower magnitude of loss in temporal degrees of freedom than `scrubbing` or `simple` strategies.
However, we did not obser this advantage in the two datasets shown in this benchmark. 
The reason potentially lies in the implementation of ICA-AROMA.
ICA-AROMA uses pretrained model on healthy subjects to select noise components {cite:p}`aroma`. 
The two datasets used in the current benchmark both contained subjects with behaviours deviating from the healthy controls. 
`ds000228` is comprised of adult healthy controls and children. 
`ds000030` includes healthy controls and subjects with three different psychiatric conditions. 

The figures showed the number of regressors used at different stage of denoising. 
The `simple` and `srubbing`-based strategy and `compcor6` are the strategy with a fixed number of degree of freedom loss.
`compcor` and `aroma`-based strategies shows variability depending on the number of noise compoenets detected.
We 

`````{tab-set}

````{tab-item} Overview
```{glue:figure} dof-fig
:figwidth: 800px
:name: "tbl:dof-fig"

Loss in temporal degrees of freedom in full sample of each dataset. 
```
````

````{tab-item} ds000228

```{glue:figure} dof-ds000228-fig
:figwidth: 800px
:name: "tbl:dof-ds000228-fig"

Loss in temporal degrees of freedom break down by groups. 
```
````

````{tab-item} ds000030
```{glue:figure} dof-ds000030-fig
:figwidth: 800px
:name: "tbl:dof-ds000030-fig"

Loss in temporal degrees of freedom break down by diagnostics. 
```
````

`````

Scrubbing

`````{tab-set}

````{tab-item} Overview
```{glue:figure} scrubbing-fig
:figwidth: 800px
:name: "tbl:scrubbing-fig"

Loss in number of volumes in full sample of each dataset. 
```
````

````{tab-item} ds000228

```{glue:figure} scrubbing-ds000228-fig
:figwidth: 800px
:name: "tbl:scrubbing-ds000228-fig"

Loss in temporal degrees of freedom break down by groups. 
```
````

````{tab-item} ds000030
```{glue:figure} scrubbing-ds000030-fig
:figwidth: 800px
:name: "tbl:scrubbing-ds000030-fig"

Loss in number of volumes break down by diagnostics. 
```
````

`````

<!-- new analysis: direct compare mean FD between different groups -->
