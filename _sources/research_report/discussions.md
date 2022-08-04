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

# Discussions

Selecting the correct conofund variables for denoising minimally preprocessed fMRI data is a challenging task.
fMRIPrep ensure a large collection of metrics are delivered in the standard way.
They provided documentation to guide the user for regressor selection.
However, there's no guarantee it is goining to be correct or people are reading the documentation properly.
`load_confounds` is an API to simplify the process.
Built with careful curation of the literature, it ensures users select the sensible nusiance regressors.
It also provide predefined strategies with the best understanding of the literature.
The remaining issue is that the performance of these denoising strategies has yet to be exaimined.
The current research aims to provide benchmark on the performance of common methods in the denoising literature for functional connectivity analysis.

We performed the benchmark on two open access datasets: ds000228 and ds000030.
ds000228 allows us to examine the difference between children and adult.
ds000030 includes healthy control and patients of three different psychiatric conditions.
We used three different multi-resolution atlases in this benchmark (MIST, DiFuMo, Schaefer)
as well as one common altlas in the previous benchmark literature,
the Gordon atlas with 333 parcels, to compare with the past literature using different pipelines.
The benchmark is largely consistent with the recent literatures.
Comparing to the baseline, data that was high-pass filtered,
<!-- some description of the results -->
However, the results related to GSR are weird in both dataset.
Further investigation is needed on the code for signal cleaning in nilearn and GSR calculation implemented in fMRIPrep.

Scrubbing based method is the best if the analysis is using summary statistics such as functional connectivity at individual level.
For analysis requires the continueous timeseries, the simple strategy might be the most balanced approach.
Aside for the benchmark on denoising performance, another important factor is the degree of freedome loss.
Losing degree of freedom means the variance left for the subsequent analysis will be limited.
Despite the good performance of ICA-AROMA, it is the method with the highest losss in degrees of freedom.
<!-- check what ciric et al say about the degree of freedom -->
For dataset with shorter data acquisition sequence, methods using too many nuisance regressors might not be a good idea.
CompCor and ICA-AROMA both performs well but can risk in lossing to much degrees of freedom.
Compcor method should be used with caution.
The six component cut-off does not respect differences of each dataset, but does perform well on average.
Our beenchmark shows the 50% variance approach doesn't perform better than just a smaller subset of the components.
And it risk reintroducing noise to the data and a higher cost in loss of degree of freedom.
Both CompCor and ICA-AROMA produce components that are approximation of likely noise.
Component based method might not be desirable for users who wish to have explicit definition for their regressors.
In summary, simple is a sufficient enough approach and scrubbing is great if network structure is a priority and timeseries property is not needed.

<!-- write some stuff about the clinical population -->

With benchmark on different type of atlas, we also show that the impact of altas selection is not big.
Volumetric data based atlas, both probablility and descret atlas performs similarly.
Surface based atlas provides less variance across different resolutions.
The current software will be maintained along with the release of fMRIprep.
The code base used in this paper provides a great foundation for generating reports on dataset for post-fMRIPrep processing.
The reports on these two dataset will be able to regenerate the for the future release of fMRIprep.
