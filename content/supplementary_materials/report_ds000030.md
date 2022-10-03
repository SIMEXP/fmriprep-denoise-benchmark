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

# ds000030

## QC-FC and distance-dependent effect

`ds000030` consists of adult sample only.
All denoisng strategies aside from `aroma+gsr` eliminate the impact of motion.
The variablilty in the healthy control is potentially driven by a larger sample than the rest.
However, when looking at the median absolute deviations, the schizophrania group still retains higher impact of motion than the remaining sample.


```{code-cell}
:tags: [hide-input, remove-output]

import warnings

warnings.filterwarnings('ignore')
from fmriprep_denoise.visualization import figures, utils
from myst_nb import glue

path_root = utils.get_data_root() / "denoise-metrics"

# Load metric data
dataset = 'ds000030'
atlases = ['mist', 'difumo', 'schaefer7networks', 'gordon333']
groups = ['CONTROL', 'SCHZ', 'BIPOLAR', 'ADHD']
for atlas in atlases:
    for group in groups:
        fig = figures.plot_motion_resid(dataset, path_root, atlas, group=group)
        glue(f'{dataset}_{group}_{atlas}_qcfc-fig', fig, display=False)
```
<!-- 
### Full sample

:::{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_full_sample_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_full_sample_mist_qcfc-fig"

Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```
````

````{tab-item} DiFuMo
```{glue:figure} ds000030_full_sample_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_full_sample_difumo_qcfc-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas.
Each data point represent different resolution.
```
````

````{tab-item} Schaefer 7 Networks
```{glue:figure} ds000030_full_sample_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_full_sample_schaefer7networks_qcfc-fig"

Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```
````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000030_full_sample_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_full_sample_gordon333_qcfc-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```
````
::: -->

:::{tab-set}

````{tab-item} MIST

```{glue:figure} ds000030_CONTROL_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_mist_qcfc-fig"

CONTROL.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_SCHZ_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_mist_qcfc-fig"

SCHZ.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_BIPOLAR_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_mist_qcfc-fig"

BIPOLAR.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_ADHD_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_mist_qcfc-fig"

ADHD.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

````

````{tab-item} DiFuMo

```{glue:figure} ds000030_CONTROL_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_difumo_qcfc-fig"

CONTROL.
Residual effect of motion on connectomes generated with DiFuMo atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_SCHZ_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_difumo_qcfc-fig"

SCHZ.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_BIPOLAR_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_difumo_qcfc-fig"

BIPOLAR.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_ADHD_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_difumo_qcfc-fig"

ADHD.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

````

````{tab-item} Schaefer 7 Networks

```{glue:figure} ds000030_CONTROL_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_schaefer7networks_qcfc-fig"

CONTROL.
Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_SCHZ_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_schaefer7networks_qcfc-fig"

SCHZ.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_BIPOLAR_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_schaefer7networks_qcfc-fig"

BIPOLAR.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_ADHD_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_schaefer7networks_qcfc-fig"

ADHD.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

````

````{tab-item} Gordon 333 parcels

```{glue:figure} ds000030_CONTROL_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_gordon333_qcfc-fig"

CONTROL.
Residual effect of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```

```{glue:figure} ds000030_SCHZ_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_gordon333_qcfc-fig"

SCHZ.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_BIPOLAR_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_gordon333_qcfc-fig"

BIPOLAR.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_ADHD_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_gordon333_qcfc-fig"

ADHD.
Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

````
:::

## Distance-dependent effects of motion on connectivity

Scrubbing with 0.2 mm threshold most consistently reduce the pairwise distance-motion dependency across different sample and atlas of choice. 
The MIST atlas  shows more negative trend than the other choices of atlas.


```{code-cell}
:tags: [hide-input, remove-output]
for atlas in atlases:
    for group in groups:
        fig = figures.plot_distance_dependence(dataset, path_root, atlas, group=group)
        glue(f'{dataset}_{group}_{atlas}_dist-fig', fig, display=False)
```
:::{tab-set}

````{tab-item} MIST

```{glue:figure} ds000030_CONTROL_mist_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_mist_dist-fig"

CONTROL.
Distance-dependent effects of motion on connectivity with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_SCHZ_mist_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_mist_dist-fig"

SCHZ.
Distance-dependent effects of motion on connectivity with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_BIPOLAR_mist_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_mist_dist-fig"

BIPOLAR.
Distance-dependent effects of motion on connectivity with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

```{glue:figure} ds000030_ADHD_mist_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_mist_dist-fig"

ADHD.
Distance-dependent effects of motion on connectivity with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
```

````

````{tab-item} DiFuMo

```{glue:figure} ds000030_CONTROL_difumo_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_difumo_dist-fig"

CONTROL.
Distance-dependent effects of motion on connectivity with DiFuMo atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_SCHZ_difumo_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_difumo_dist-fig"

SCHZ.
Distance-dependent effects of motion on connectivity with DiFuMo atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_BIPOLAR_difumo_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_difumo_dist-fig"

BIPOLAR.
Distance-dependent effects of motion on connectivity with DiFuMo atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_ADHD_difumo_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_difumo_dist-fig"

ADHD.
Distance-dependent effects of motion on connectivity with DiFuMo atlas.
Each data point represent different resolution.
```

````

````{tab-item} Schaefer 7 Networks

```{glue:figure} ds000030_CONTROL_schaefer7networks_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_schaefer7networks_dist-fig"

CONTROL.
Distance-dependent effects of motion on connectivity with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_SCHZ_schaefer7networks_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_schaefer7networks_dist-fig"

SCHZ.
Distance-dependent effects of motion on connectivity with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_BIPOLAR_schaefer7networks_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_schaefer7networks_dist-fig"

BIPOLAR.
Distance-dependent effects of motion on connectivity with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

```{glue:figure} ds000030_ADHD_schaefer7networks_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_schaefer7networks_dist-fig"

ADHD.
Distance-dependent effects of motion on connectivity with Schaefer 7 Networks atlas.
Each data point represent different resolution.
```

````

````{tab-item} Gordon 333 parcels

```{glue:figure} ds000030_CONTROL_gordon333_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_gordon333_dist-fig"

CONTROL.
Distance-dependent effects of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```

```{glue:figure} ds000030_SCHZ_gordon333_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_gordon333_dist-fig"

SCHZ.
Distance-dependent effects of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```

```{glue:figure} ds000030_BIPOLAR_gordon333_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_gordon333_dist-fig"

BIPOLAR.
Distance-dependent effects of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```

```{glue:figure} ds000030_ADHD_gordon333_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_gordon333_dist-fig"

ADHD.
Distance-dependent effects of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
```

````
:::


## Network modularity

Results here are so fuzzy I don't know what to do.

```{code-cell}
:tags: [hide-input, remove-output]

import warnings

warnings.filterwarnings('ignore')
from fmriprep_denoise.visualization import figures
from myst_nb import glue

# Load metric data
dataset = 'ds000030'
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

<!-- ```{glue:figure} ds000030_mist_network-fig
:figwidth: 800px
:name: "tbl:ds000030_mist_network-fig"

Residual effect of motion on connectomes generated with MIST atlas.
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The 7, 12, 20 ROIs version of the atlas were excluded from plotting as these versions drives the outliers in these measures.
``` -->

```{glue:figure} ds000030_mist_network-1-fig
:figwidth: 800px
:name: "tbl:ds000030_mist_network-1-fig"

```

```{glue:figure} ds000030_mist_network-2-fig
:figwidth: 800px
:name: "tbl:ds000030_mist_network-2-fig"

```

```{glue:figure} ds000030_mist_network-3-fig
:figwidth: 800px
:name: "tbl:ds000030_mist_network-3-fig"

```

```{glue:figure} ds000030_mist_network-4-fig
:figwidth: 800px
:name: "tbl:ds000030_mist_network-4-fig"

```

````

````{tab-item} DiFuMo

<!-- ```{glue:figure} ds000030_difumo_network-fig
:figwidth: 800px
:name: "tbl:ds000030_difumo_network-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas.
Each data point represent different resolution.
``` -->

```{glue:figure} ds000030_difumo_network-1-fig
:figwidth: 800px
:name: "tbl:ds000030_difumo_network-1-fig"

```

```{glue:figure} ds000030_difumo_network-2-fig
:figwidth: 800px
:name: "tbl:ds000030_difumo_network-2-fig"

```

```{glue:figure} ds000030_difumo_network-3-fig
:figwidth: 800px
:name: "tbl:ds000030_difumo_network-3-fig"

```

```{glue:figure} ds000030_difumo_network-4-fig
:figwidth: 800px
:name: "tbl:ds000030_difumo_network-4-fig"

```

````

````{tab-item} Schaefer 7 Networks

<!-- ```{glue:figure} ds000030_schaefer7networks_network-fig
:figwidth: 800px
:name: "tbl:ds000030_schaefer7networks_network-fig"

Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas.
Each data point represent different resolution.
``` -->

```{glue:figure} ds000030_schaefer7networks_network-1-fig
:figwidth: 800px
:name: "tbl:ds000030_schaefer7networks_network-1-fig"

```

```{glue:figure} ds000030_schaefer7networks_network-2-fig
:figwidth: 800px
:name: "tbl:ds000030_schaefer7networks_network-2-fig"

```

```{glue:figure} ds000030_schaefer7networks_network-3-fig
:figwidth: 800px
:name: "tbl:ds000030_schaefer7networks_network-3-fig"

```

```{glue:figure} ds000030_schaefer7networks_network-4-fig
:figwidth: 800px
:name: "tbl:ds000030_schaefer7networks_network-4-fig"

```

````

````{tab-item} Gordon 333 parcels

<!-- ```{glue:figure} ds000030_gordon333_network-fig
:figwidth: 800px
:name: "tbl:ds000030_gordon333_network-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels.
In this atlas there's only one parcellation scheme.
``` -->

```{glue:figure} ds000030_gordon333_network-1-fig
:figwidth: 800px
:name: "tbl:ds000030_gordon333_network-1-fig"

```

```{glue:figure} ds000030_gordon333_network-2-fig
:figwidth: 800px
:name: "tbl:ds000030_gordon333_network-2-fig"

```

```{glue:figure} ds000030_gordon333_network-3-fig
:figwidth: 800px
:name: "tbl:ds000030_gordon333_network-3-fig"

```

```{glue:figure} ds000030_gordon333_network-4-fig
:figwidth: 800px
:name: "tbl:ds000030_gordon333_network-4-fig"

```

````
:::
