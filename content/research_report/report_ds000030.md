---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# ds000030

## QC-FC and distance-dependent effect
<!-- 
No denoise strategy removed the correlation with motion captured by mean framewise displacement. 
`aroma`, `compcor6`, and `simple` reduced the correlation between connectivity edges and mean framewise displacement. 
`scrubbing` and `scrubbing+gsr` performed the best, as seen in previous work {cite:p}`power_recent_2015`. 
`compcor`, which applies compcor-based regressors covering 50% of the variance, performs worse than the connectome baseline created with high-pass filtered time series. 
Surprisingly, all strategies with global signal regression underperform, contradicting the existing literature {cite:p}`ciric_benchmarking_2017` {cite:p}`parkes_evaluation_2018`. -->

```{code-cell} python3
:tags: [hide-input, remove-output]
import warnings
warnings.filterwarnings("ignore")
from fmriprep_denoise.visualization import figures
from myst_nb import glue

# Load metric data
dataset = "ds000030"
atlases = ["mist", "difumo", "schaefer7networks", "gordon333"]
groups = ['full_sample', 'CONTROL', 'SCHZ', 'BIPOLAR', 'ADHD']
for atlas in atlases:
  for group in groups:
    fig = figures.plot_motion_resid(dataset, atlas, group=group)
    glue(f"{dataset}_{group}_{atlas}_qcfc-fig", fig, display=False)
```

### Full sample

`````{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_full_sample_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_full_sample_mist_qcfc-fig"

Residual effect of motion on connectomes generated with MIST atlas. 
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The outliers here are driven by the 7, 12, 20 ROIs version of the atlas.
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
`````

### Healthy control 

`````{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_CONTROL_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_mist_qcfc-fig"

Residual effect of motion on connectomes generated with MIST atlas. 
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The outliers here are driven by the 7, 12, 20 ROIs version of the atlas.
```
````

````{tab-item} DiFuMo
```{glue:figure} ds000030_CONTROL_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_difumo_qcfc-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Schaefer 7 Networks
```{glue:figure} ds000030_CONTROL_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_schaefer7networks_qcfc-fig"

Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000030_CONTROL_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_CONTROL_gordon333_qcfc-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels. 
In this atlas there's only one parcellation scheme.
```
````
`````

### SCHZ 

`````{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_SCHZ_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_mist_qcfc-fig"

Residual effect of motion on connectomes generated with MIST atlas. 
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The outliers here are driven by the 7, 12, 20 ROIs version of the atlas.
```
````

````{tab-item} DiFuMo
```{glue:figure} ds000030_SCHZ_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_difumo_qcfc-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Schaefer 7 Networks
```{glue:figure} ds000030_SCHZ_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_schaefer7networks_qcfc-fig"

Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000030_SCHZ_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_SCHZ_gordon333_qcfc-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels. 
In this atlas there's only one parcellation scheme.
```
````
`````

### BIPOLAR 

`````{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_BIPOLAR_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_mist_qcfc-fig"

Residual effect of motion on connectomes generated with MIST atlas. 
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The outliers here are driven by the 7, 12, 20 ROIs version of the atlas.
```
````

````{tab-item} DiFuMo
```{glue:figure} ds000030_BIPOLAR_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_difumo_qcfc-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Schaefer 7 Networks
```{glue:figure} ds000030_BIPOLAR_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_schaefer7networks_qcfc-fig"

Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000030_BIPOLAR_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_BIPOLAR_gordon333_qcfc-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels. 
In this atlas there's only one parcellation scheme.
```
````
`````


### ADHD 

`````{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_ADHD_mist_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_mist_qcfc-fig"

Residual effect of motion on connectomes generated with MIST atlas. 
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The outliers here are driven by the 7, 12, 20 ROIs version of the atlas.
```
````

````{tab-item} DiFuMo
```{glue:figure} ds000030_ADHD_difumo_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_difumo_qcfc-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Schaefer 7 Networks
```{glue:figure} ds000030_ADHD_schaefer7networks_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_schaefer7networks_qcfc-fig"

Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000030_ADHD_gordon333_qcfc-fig
:figwidth: 800px
:name: "tbl:ds000030_ADHD_gordon333_qcfc-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels. 
In this atlas there's only one parcellation scheme.
```
````
`````


## Distance-dependent effects of motion on connectivity

<!-- Consistent with the literature, `aroma` reduces the distance dependency of motion on connectivity.  -->

```{code-cell} python3
:tags: [hide-input, remove-output]
import warnings
warnings.filterwarnings("ignore")
from fmriprep_denoise.visualization import figures
from myst_nb import glue

# Load metric data
dataset = "ds000030"
atlases = ["difumo", "mist", "schaefer7networks", "gordon333"]
for atlas in atlases:
  fig = figures.plot_distance_dependence(dataset, atlas)
  glue(f"{dataset}_{atlas}_dist-fig", fig, display=False)
```

`````{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_mist_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_mist_dist-fig"

Distance-dependent effects of motion on connectivity with MIST atlas. 
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The outliers here are driven by the 7, 12, 20 ROIs version of the atlas.
```
````

````{tab-item} DiFuMo
```{glue:figure} ds000030_difumo_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_difumo_dist-fig"

TBA
```
````

````{tab-item} Schaefer 7 Networks
```{glue:figure} ds000030_schaefer7networks_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_schaefer7networks_dist-fig"

Distance-dependent effects of motion on connectivity with Schaefer 7 Networks atlas. 
Each data point represent different resolution.
```
````

````{tab-item} Gordon 333 parcels
```{glue:figure} ds000030_gordon333_dist-fig
:figwidth: 800px
:name: "tbl:ds000030_gordon333_dist-fig"

Distance-dependent effects of motion on connectivity with Gordon atlas 333 parcels. 
In this atlas there's only one parcellation scheme.
```
````
`````


## Network modularity
<!-- 
All strategies increased the overall network modularity compared to the `baseline`, with scrubbing based methods performing the best out of all. 
GSR-based strategies improved the network modularity compared to their conunterparts.
The correlation between modularity quality and motion for each denoising approach shows that compcor-based and ICA-AROMA strategies are the best at eliminating correlations between motion and modularity. -->

```{code-cell} python3
:tags: [hide-input, remove-output]
import warnings
warnings.filterwarnings("ignore")
from fmriprep_denoise.visualization import figures
from myst_nb import glue

# Load metric data
dataset = "ds000030"
atlases = ["mist", "difumo", "schaefer7networks", "gordon333"]
for atlas in atlases:
  fig = figures.plot_network_modularity(dataset, atlas, by_group=False)
  glue(f"{dataset}_{atlas}_network-fig", fig, display=False)
  figs = figures.plot_network_modularity(dataset, atlas, by_group=True)
  for i, fig in enumerate(figs):
      glue(f"{dataset}_{atlas}_network-{i+1}-fig", fig, display=False)
```


`````{tab-set}
````{tab-item} MIST
```{glue:figure} ds000030_mist_network-fig
:figwidth: 800px
:name: "tbl:ds000030_mist_network-fig"

Residual effect of motion on connectomes generated with MIST atlas. 
Each data point represent different resolution.
MIST atlas includes some coarse parcels (< 64 ROIs) that are not practical for compression.
The outliers here are driven by the 7, 12, 20 ROIs version of the atlas.
```

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
```{glue:figure} ds000030_difumo_network-fig
:figwidth: 800px
:name: "tbl:ds000030_difumo_network-fig"

Residual effect of motion on connectomes generated with DiFuMo atlas. 
Each data point represent different resolution.
```

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
```{glue:figure} ds000030_schaefer7networks_network-fig
:figwidth: 800px
:name: "tbl:ds000030_schaefer7networks_network-fig"

Residual effect of motion on connectomes generated with Schaefer 7 Networks atlas. 
Each data point represent different resolution.
```

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
```{glue:figure} ds000030_gordon333_network-fig
:figwidth: 800px
:name: "tbl:ds000030_gordon333_network-fig"

Residual effect of motion on connectomes generated with Gordon atlas 333 parcels. 
In this atlas there's only one parcellation scheme.
```

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
`````

:::{dropdown} References on this page

```{bibliography}
:filter: docname in docnames
```
:::