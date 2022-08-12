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

<!-- Take home message
- fMRIPrep does a good job 
- Quality control your data
- Choise of atlas doesn't impact the results much
- ICA-AROMA is not magical
 -->

fMRIPrep ensured to provide minimally preprocessed fMRI data and a large collection of nuisance regressors in the standard way.
They provided extended documentation to guide the user for regressor selection and choices for subsequent analysis.
However, selecting the correct conofund variables for denoising minimally preprocessed fMRI data is still a daunting task.
There's no guarantee that people are reading the documentation to select the right things. <!-- This sentence is too blunt and harsh -->
The aim of `load_confounds` is to provide an API to simplify the process of nuisance regressors selection.
Built with careful curation of the literature, it ensures users select the sensible nusiance regressors.
It also provide predefined strategies with the best understanding of the literature.
The performance of these denoising strategies has yet to be exaimined.
The current research aims to provide benchmark on the performance of common methods in the denoising literature for functional connectivity analysis.
We only includes denoise strategies that are realistically applicable to the sample.
In addition, we set some exclusion criteria on the sample to show the result on reasonable quality subject only.
The aim is to provide a benchmark in a realistic context as much as possible.

We performed the benchmark on two open access datasets: `ds000228` and `ds000030`.
`ds000228` allows us to examine the difference between child and adult sample.
`ds000030` includes healthy control and patients of three different psychiatric conditions.
We used three different multi-resolution atlases in this benchmark (MIST, DiFuMo, Schaefer)
as well as one common altlas in the previous benchmark literature,
the Gordon atlas with 333 parcels, to compare with the past literature using different pipelines.
Subject with more than 80% of volumes with a framewise displacement exeeding 0.2 mm, 
and mean framewise displacement above 0.55 mm were excluded from the analysis.

When examining QC/FC, all strategies, including the baseline (i.e. data that was high-pass filtered),
shows very little impact of residual imapct of motion on functional connectivity, with the exception of `aroma+gsr`.
`aroma+gsr` did not perfrom well in the healthy control in `ds000030` and the child sample in `ds000228`.
The healthy control in `ds000030` shows more variablilty possibly due to the larger sample size (88 subjects) comparing to the patient samples (ranging from 19 -- 32 subjects).
Similarly, `ds000228` contains twice as many child sample comapring to the adult sample (51 vs 24 subjects).
There's a possibility of global signal regressor reintroduce motion to the data.
In fMRIPrep, the whole brain global signal regressor and the estimated head-motion parameters was calcuated on the output from their regular pipeline (i.e. before denoising).
The `aroma` strategy perfroms as expected, which is consistent with the [simulation shown in fMRIPrep documentation](https://github.com/nipreps/fmriprep-notebooks/blob/9933a628dfb759dc73e61701c144d67898b92de0/05%20-%20Discussion%20AROMA%20confounds%20-%20issue-817%20%5BJ.%20Kent%5D.ipynb).
The global signal regressor might reintroduce motion. 
This would be consistent with the suggestion that confound regressors should be recalculated when using AROMA-cleaned data in the literature {cite:p}`hallquist_nuisance_2013,lindquist_modular_2019`.

<!-- For the distance depnendency affect on QC/FC, we found -->
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
