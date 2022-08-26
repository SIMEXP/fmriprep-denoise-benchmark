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
- Choice of atlas doesn't impact the results much
- ICA-AROMA is not magical
 -->

fMRIPrep ensured to provide minimally preprocessed fMRI data and a large collection of nuisance regressors in the standard way.
They provided extended documentation to guide the user for regressor selection and choices for subsequent analysis.
However, selecting the correct confound variables for denoising minimally preprocessed fMRI data is still a daunting task.
There's no guarantee that people are reading the documentation to select the right things. <!-- This sentence is too blunt and harsh -->
`load_confounds` aims to provide an API to simplify the process of nuisance regressors selection.
Built with careful curation of the literature, it ensures users select the sensible nuisance regressors.
It also provides predefined strategies with the best understanding of the literature.
The performance of these denoising strategies has yet to be examined.
The current research aims to provide a benchmark on the performance of common methods in the denoising literature for functional connectivity analysis.
We only include denoise strategies that apply to the sample.
In addition, we set some exclusion criteria on the sample to show the result on reasonable quality subjects only.
The aim is to provide a benchmark in a realistic context as much as possible.

We performed the benchmark on two open access datasets.
`ds000228` allows us to examine the difference between child and adult samples.
`ds000030` includes healthy control and patients of three different psychiatric conditions.
We used three different multi-resolution atlases in this benchmark (MIST, DiFuMo, Schaefer)
as well as one common atlas in the previous benchmark literature,
the Gordon atlas with 333 parcels, to compare with the past literature using different pipelines.
Subjects with more than 80% of volumes with a framewise displacement exceeding 0.2 mm, 
and mean framewise displacement above 0.55 mm were excluded from the analysis.

## Residual motion in connectivity edges was not found among most methods
<!-- QC-FC -->
When examining QC-FC, all strategies, including the baseline (i.e. data that was high-pass filtered),
show very little residual impact of motion on functional connectivity, except `aroma+gsr`.
`aroma+gsr` did not perform well in the healthy control in `ds000030` and the child sample in `ds000228`.
The healthy control in `ds000030` shows more variability possibly due to the larger sample size (88 subjects) compared to the patient samples (ranging from 19 -- 32 subjects).
Similarly, `ds000228` contains twice as many child samples compared to the adult samples (51 vs 24 subjects).
There's a possibility of the global signal regressor reintroducing motion to the data.
In fMRIPrep, the whole brain global signal regressor and the estimated head-motion parameters were calculated on the output from their regular pipeline (i.e. before denoising).
The `aroma` strategy performs as expected, which is consistent with the [simulation shown in fMRIPrep documentation](https://github.com/nipreps/fmriprep-notebooks/blob/9933a628dfb759dc73e61701c144d67898b92de0/05%20-%20Discussion%20AROMA%20confounds%20-%20issue-817%20%5BJ.%20Kent%5D.ipynb).
The global signal regressor might reintroduce motion. 
This would be consistent with the suggestion that confound regressors should be recalculated when using AROMA-cleaned data in the literature {cite:p}`hallquist_nuisance_2013,lindquist_modular_2019`.

## Distance dependency of connectivity edges was best removed by an aggressive scrubbing approach
<!-- Distance dependency -->
For the distance dependency effect on QC-FC, a score close to zero indicates a method to reduce the correlation between residual motion in functional connectivity and pairwise node distance. 
We found scrubbing with a 0.2 mm threshold mitigates distance dependency well consistently regardless of the type of subject. 
DiFuMo on adult subjects across both datasets shows consistent results in reducing the impact of motion on distance dependency.

## Correlation between motion and network modularity persists and global signal alters network modularity
<!-- network modularity -->
To quantify the impact of confound removal on meaningful signals, we evaluated the modularity quality of the connectomes. 
The correlation between motion and modularity was not eliminated by any methods. 
For the mean modularity score, we can see two clear cluster grouping methods separated by including the global signal regressor or not. 
Including global signal regressor increases the network modularity in `simple+gsr`, `scrubbing.5+gsr`, `scrubbing.2+gsr`, and `aroma+gsr` (with a smaller magnitude). 
The result is consistent with the fact that global signal increases negative values to functional connectome 
(see nilearn examples visualisation of connectome [with and without global signal regression](https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_signal_extraction.html#the-impact-of-global-signal-removal))
by shifting the value to approximately zero-centred {cite:p}`murphy_gsr_2009`. 

## High degrees of freedom loss with using CompCor components up to 50% variance
<!-- dof loss -->
Aside from the benchmark on denoising performance, another important factor is the degree of freedom loss. 
Losing degrees of freedom means the variance left for the subsequent analysis will be limited. 
The default CompCor output of fMRIPrep (the 50% variance) resulted in the highest number of regressors used during denoising. 
Contrary to the previous literature {cite:p}`parkes_evaluation_2018` we do not see much advantage of ICA-AROMA.
The independent components put the number of regressors in a similar range to the `simple` and `compcor6` strategies.
Scrubbing at 0.5 mm or 0.2 mm shows a difference between strategies; 
within each dataset, this difference persists across groups.

## Other considerations
<!-- other things to consider -->

We included different types of the atlas in the current analysis. 
We got clear outliers with the MIST atlas including decompositions of a low number of parcels. 
For any atlas with 64 or above parcels, we show that atlas selection did not bring critical differences to the benchmark. 
Volumetric data-based atlas, both probability and discrete atlas perform similarly. 
Surface-based Schaefer atlas provides less variance across different resolutions.

We showed the difference between the child and adult samples cannot be eliminated by including age as a covariate in `ds000228`.
Moreover, the loss of degrees of freedom shows that two samples require a drastically different number of regressors for `compcor` and `aroma`-based methods.
`simple` might be the better strategy if the inconsistency of loss of degrees of freedom is not desired. 
For the clinical groups in `ds000030`, we only see differences between the control and the schizophrenia group in gross framewise displacement.
The remaining results are too fussy to make a reliable observation amongst the clinical groups.
The limitation of the current results lies in the small and uneven subgroups in each dataset. 
The small sample size constrains the conclusion we can draw from the comparison.

## General suggestions

We speculate that the differences highlighted in the literature could be a result of including strawman methods that are not applicable or commonly used in a practical research setting  (see 2P, tCompCor in {cite}`ciric_benchmarking_2017`), 
and including known high motion subjects (children) with young adult sample 
(see {cite}`ciric_benchmarking_2017` including 8 --22 years old in the same analysis). <!-- please help me with more diplomatic way of saying this -->
We performed partial correlation for all correlation-based metrics to address the covariate introduced by age.
However, the differences between adult and child samples persist.
With real-data applicable methods and motion-based data exclusion, combining all metrics evaluated, 
we find it difficult to conclude the best method for different types of subjects. 
ICA-AROMA did not stand out in our benchmark as the previous benchmarks highlighted {cite:p}`ciric_benchmarking_2017,parkes_evaluation_2018`.

Although the data-driven methods were invented to solve the loss in degrees of freedom for the head motion-based strategy,
we did not observe a clear advantage. 
There are a few data-driven strategies that researchers should adopt with caution when using fMRIPrep outputs.
For the CompCor-based strategy, the 50% variance approach `compcor` performed comparably to the 6 component strategy `compcor6` with a significant loss in degrees of freedom. 
`aroma+gsr` produces concerning results on the QC/FC measures being the only method with an inconsistent amount of edgest correlating with mean framewise displacement. 
The performance of `scrubbing.5` is similar to `simple`, 
suggesting that a loose scrubbing threshold might not bring many advantages compared to the more stringent threshold and at risk of losing degrees of freedom.
For the remaining strategies, we would suggest researchers apply denoising strategies that best fit their needs.
Data-driven, component-based methods (`aroma`, `compcor`, and `compcor6`) can be applied to data with a high number of volumes. 
We still recommend users investigate the ratio of the number of regressors to the full length of the scan.
They might not be desirable for users who wish to have an explicit definition for their regressors. 
For users who wish to have an explicit and simple definition of the nuisance regressors, `simple` is a sufficient approach, and `scrubbing.2` is great if network structure is a priority and timeseries property is not needed. 
The variation of any methods involving  global signal regressors should be applied with caution as it highlights the network property of the data (for more on this topic, please see {cite}`saad_gsr_2012,murphy_fox_2017`)

## Future directions

:::{margin}
```{admonition} fMRIPrep long-term support (LTS) release
:class: tip

Read the [fMRIPrep LTS 20.2.x blog post](https://reproducibility.stanford.edu/fmriprep-lts/) 
to understand more details about the long-term support model and the relevance for reproducibility.
```
:::

<!-- send help PLZ -->

The current work will serve as a benchmark for fMRIPrep users when deciding on the denoising strategy for their analysis workflow. 
The confounds variable selection standardised through `load_confounds`, 
we showed all regressors generated by fMRIprep perform better than the comparison baseline, except for `aroma+gsr`. 
We hope this paper presents a combining case that with the maturity of fMRI preprocessing software (fMRIPrep), 
the common denoising strategies are achieving the goal and the choice of strategies is down to the researchers. 
There is no definitive best option for denoising but methods that fit the different purposes.

For the software aspect, with the long-term support (LTS) release of fMRIPrep, the results reported here will be applicable for 20.2.x series up to September 2024. 
The API `load_confounds` is implemented in `nilearn` and maintained with community feedback. We completed the work with all elements under an open source license, including data sets, software, and code. 
Combining with the three elements above, 
the reports on these two datasets will be able to regenerate for the future LTS release of fMRIprep. 
For fMRIPrep release beyond the LTS version, as long as the API in `nilearn` is maintained, 
the code used to generate all current reports can be applied to the same two datasets. 
Thus, rebuilding this paper on future fMRIPrep releases can be a potential for verifying the stability of preprocessed results at the dataset level, 
complementing the individual reports from fMRIPrep.