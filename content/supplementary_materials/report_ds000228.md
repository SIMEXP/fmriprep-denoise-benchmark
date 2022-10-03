---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# ds000228

## QC-FC and distance-dependent effect

With good quality data, all denoising methonds reduce the corelation between functional connectivity and mean framewise displacement.
All strategies, including the baseline, 
shows motion remains in close to 0% of the connectivity edges.
`aroma+gsr` performs worse than the baseline in the child sample.
This result is consistent across atlases of choice.
The meduan absolute deviation of QC-FC are all similar to the baseline. 


```{code-cell}
:tags: [hide-input, remove-output]

import warnings

warnings.filterwarnings('ignore')
from fmriprep_denoise.visualization import figures, utils
from myst_nb import glue

path_root = utils.get_data_root() / "denoise-metrics"

# Load metric data
dataset = 'ds000228'
atlases = ['mist', 'difumo', 'schaefer7networks', 'gordon333']
groups = ['adult', 'child']
for atlas in atlases:
    for group in groups:
        fig = figures.plot_motion_resid(dataset, path_root, atlas, group=group)
        glue(f'{dataset}_{group}_{atlas}_qcfc-fig', fig, display=False)
```

:::{tab-set}
````{tab-item} MIST

```{glue:figure} ds000228_child_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_child_mist_qcfc-fig"

Child.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000228_adult_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_adult_mist_qcfc-fig"

Adult. 
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

````

````{tab-item} DiFuMo
```{glue:figure} ds000228_child_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_child_difumo_qcfc-fig"

Child.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000228_adult_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_adult_difumo_qcfc-fig"

Adult.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```
````

````{tab-item} Schaefer 7 Networks
```{glue:figure} ds000228_child_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_child_schaefer7networks_child_qcfc-fig"

Child.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000228_adult_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_adult_schaefer7networks_qcfc-fig"

Adult.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```
````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000228_child_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_child_gordon333_qcfc-fig"

Child.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000228_adult_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000228_adult_gordon333_qcfc-fig"

Adult.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```
````
:::

## Distance-dependent effects of motion on connectivity

For the remaining distance-dependent effects of motion on connectivity, 
a value closer to zero reproesents better performance. 
Here we can see the distinct difference betwenn adult and child sample.
No denoising strategy stands out in the adult sample.
For the child sample, we see a general negative correlation between pair-wise distance and QC-FC. 
Scrubbing with a 0.2 mm threshold improves the measure the most, and all other strategies perfroms better than the baseline with `aroma` based strategies perfroming slightly better.


```{code-cell}
:tags: [hide-input, remove-output]
for atlas in atlases:
    for group in groups:
        fig = figures.plot_distance_dependence(dataset, path_root, atlas, group=group)
        glue(f'{dataset}_{group}_{atlas}_dist-fig', fig, display=False)
```

:::{tab-set}
````{tab-item} MIST

```{glue:figure} ds000228_adult_mist_dist-fig
:figwidth: 800px
:name: "tbl:ds000228_adult_mist_dist-fig"

Adults.
Distance-dependent effects of motion on connectivity with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000228_child_mist_dist-fig
:figwidth: 800px
:name: "tbl:ds000228_child_mist_dist-fig"

Children.
Distance-dependent effects of motion on connectivity with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

````

````{tab-item} DiFuMo

```{glue:figure} ds000228_adult_difumo_dist-fig
:figwidth: 800px
:name: "tbl:ds000228_adult_difumo_dist-fig"

Adult.
TBA
```
```{glue:figure} ds000228_child_difumo_dist-fig
:figwidth: 800px
:name: "tbl:ds000228_child_difumo_dist-fig"

Children.
TBA
````

````{tab-item} Schaefer 7 Networks

```{glue:figure} ds000228_adult_schaefer7networks_dist-fig
:figwidth: 800px
:name: "tbl:ds000228_adult_schaefer7networks_dist-fig"

Adult.
Distance-dependent effects of motion on connectivity with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000228_child_schaefer7networks_dist-fig
:figwidth: 800px
:name: "tbl:ds000228_child_schaefer7networks_dist-fig"

Childredn.
Distance-dependent effects of motion on connectivity with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000228_adult_gordon333_dist-fig
:figwidth: 800px
:name: "tbl:ds00022_adult8_gordon333_dist-fig"

Adult.
Distance-dependent effects of motion on connectivity with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```
```{glue:figure} ds000228_child_gordon333_dist-fig
:figwidth: 800px
:name: "tbl:ds000228_child  _gordon333_dist-fig"

Childredn.
Distance-dependent effects of motion on connectivity with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```

````
:::

## Network modularity

We next see if the denoising strategies can dissociate network modularity and motion. 
The correlation between modularity quality and motion for each denoising approach shows that `aroma` is the best at eliminating correlations between motion and modularity in the child sample.
The results in tha adult sample is more fuzzy.
With the mean network modularity, 
GSR-based strategies improved the network modularity compared to their conunterparts.


```{code-cell}
:tags: [hide-input, remove-output]

import warnings

warnings.filterwarnings('ignore')
from fmriprep_denoise.visualization import figures
from myst_nb import glue

# Load metric data
dataset = 'ds000228'
atlases = ['mist', 'difumo', 'schaefer7networks', 'gordon333']
for atlas in atlases:
    fig = figures.plot_network_modularity(dataset, path_root, atlas, by_group=False)
    glue(f'{dataset}_{atlas}_network-fig', fig, display=False)
    figs = figures.plot_network_modularity(dataset, path_root, atlas, by_group=True)
    for i, fig in enumerate(figs):
        glue(f'{dataset}_{atlas}_network-{i+1}-fig', fig, display=False)
```

:::{tab-set}
````{tab-item} MIST

```{glue:figure} ds000228_mist_network-1-fig
:figwidth: 800px
:name: "tbl:ds000228_mist_network-1-fig"

Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.

```

```{glue:figure} ds000228_mist_network-2-fig
:figwidth: 800px
:name: "tbl:ds000228_mist_network-2-fig"

```

````

````{tab-item} DiFuMo


```{glue:figure} ds000228_difumo_network-1-fig
:figwidth: 800px
:name: "tbl:ds000228_difumo_network-1-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000228_difumo_network-2-fig
:figwidth: 800px
:name: "tbl:ds000228_difumo_network-2-fig"

```

````

````{tab-item} Schaefer 7 Networks

```{glue:figure} ds000228_schaefer7networks_network-1-fig
:figwidth: 800px
:name: "tbl:ds000228_schaefer7networks_network-1-fig"


Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000228_schaefer7networks_network-2-fig
:figwidth: 800px
:name: "tbl:ds000228_schaefer7networks_network-2-fig"

```

````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000228_gordon333_network-fig
:figwidth: 800px
:name: "tbl:ds000228_gordon333_network-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```

```{glue:figure} ds000228_gordon333_network-1-fig
:figwidth: 800px
:name: "tbl:ds000228_gordon333_network-1-fig"


```

```{glue:figure} ds000228_gordon333_network-2-fig
:figwidth: 800px
:name: "tbl:ds000228_gordon333_network-2-fig"


```

````
:::

