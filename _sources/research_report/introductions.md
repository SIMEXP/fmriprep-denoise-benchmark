# Introductions

<!-- We all know denosing is important now. And we cannot understand brain activity without it. -->
Amongst functional magnetic resonance imaging (fMRI) acquisition protocols,
resting state fMRI has become a tool of choice to explore human brain function in healthy and diseased populations {cite:p}`biswal_2010`.
Functional connectivity {cite:p}`sporns_organization_2004`,
a measure derived from resting state fMRI, is in particular a popular method to characterize the neural basis of <!-- any suggestions of some newer paper to cite is good. NC: Changed wording here due to repetition of 'choice', but you may prefer a different word to method -->
cognitive function {cite:p}`finn_functional_2015`,
aging {cite:p}`andrews-hanna_disruption_2007`,
and pathology {cite:p}`woodward_resting-state_2015`.
The rise in popularity of resting state fMRI has also highlighted challenges and gaps in the analytic approach {cite:p}`cole_advances_2010`,
such as distortion introduced by non-neuronal sources.
These include head motion, scanner noise, cardiac and repiratory artifacts {cite:p`rogers_assessing_2007,murphy_2013`.
Known as confounds or nuisance regressors, they pose a major challenge in functional connectivity analysis {cite:p`satterthwaite_impact_2012`.
For example, when functional connectivity is contaminated with motion, it can lead to altered network structure {cite:p}`power_scrubbing_2012`.
Indeed, in younger subjects and neurologic/psychiatric patients the impact of head motion on connectivity is frequently reported {cite:p}`makowski_head_2019,satterthwaite_impact_2012`.
Hence, reducing the impact of confounds, known as denoising,
is a major component of the fMRI workflow to ensure the quality of analysis.

<!-- Classes of nuisance regressors - like how load_confounds separate them -->
<!-- need to add reference to this section-->
## Types of noise regressors

<!-- NC: I thought the opening and closing sentences of this section were repeating the same informtion unecessarily -->
<!-- so I combined them. It could also work at the end, but I think this way flows better to the next section -->
The most common method to minimize the impact of confounds in functional connectivity analysis is to perform a linear regression {cite:p}`friston_statistical_1994`.
After basic processing steps (see fMRIPrep {cite:p}`fmriprep1`), 
regressors are regressed out from the signal and the resulting residual used as denoised signal in all subsequent analyses.
Nuisance regressors can be separated into classes, capturing different types of non-neuronal noise.
__Temporal high-pass filtering__ accounts for low-frequency signal drifts introduced by _physiological and scanner noise sources_.
__Head motion__ is captured by _6 rigid-body motion parameters_ (3 translations and 3 rotation) estimated relative to a reference image {cite:p}`friston_movement-related_1996`.
__Non-grey matter tissue signal__ (such as _white matter_ and _cerebrospinal fluid_), unlikely to reflect neuronal activity {cite:p}`fox_pnas_2005`,
is captured by averaging signal within anatomically-derived masks.
__Global signal__ is calculated by averaging signal within the _full brain_ mask {cite:p}`fox_pnas_2005`.

These four classes of regressors can be expanded to their first temporal derivatives and their quadratic terms {cite:p}`satterthwaite_2013`.
The principle component based method __CompCor__ {cite:p}`behzadi_compcor_2007` extracts principal components from white matter and cerebrospinal fluid masks to estimate non-neuronal activity.
Independent component analysis based methods, __ICA-FIX__ {cite:p}`salimi-khorshidi_automatic_2014` and __ICA-AROMA__ {cite:p}`aroma`,
estimate independent component time series related to head-motion through a data-driven classifier (ICA-FIX) or a pre-trained model (ICA-AROMA).
Different strategies have particular strengths,
and an optimal approach often involves combining a few classes of regressors described above.

## Implementation of denoising step

<!-- How denoising is traditionally done in propriatory software -->
<!-- NC: changed this to present tense as I assume people still use the software being described? -->
Which approach to adopt is straightforward when users use preprocessing software with statistical modelling functionality.
Each software implements their own preferred strategy,
for instance, FSL adds the motion parameters by default, and provides an option to run ICA-AROMA.
The noise regressors are preselected,
and thus user intervention is avoided.
However, when users wish to use additional regressors, the process can be complicated.
With the recent benchmark papers on confound denoising, <!-- NC: references here?  -->
it's not uncommon for users to want to add parameters other than the native options from a given software.
In the FSL example, if a user wants to use an unsupported strategy, such as CompCor,
they will have to calculate the regressors and inject into the workflow manually.
<!-- Question: should we compare some other software? ie. niak and cpac has a more flexible approach, but still lock user-in  -->
<!-- NC: sounds relevant to me! Could be just one sentence  -->

The recent rise of minimal preprocessing pipelines fMRIPrep {cite:p}`fmriprep1` has addressed the confound regressor issue from a different angle.
fMRIPrep aims to standardize the established steps of fMRI preprocessing, including registration, slice timing correction, motion correction, and distortion correction.
Their approach leaves the choice of denoising and spatial smoothing to users.
Instead of generating a small subset of confound regressors, fMRIPrep calculates a wide range of regressors that can be extracted from the fMRI data.
The benefit of this approach is that the generation of these regressors is standardised and accessible from the same source.
However, several issues related to how users select the regressors for the denoising step remain.
A common concern is that users unfamiliar with the denoising literature or the fMRIprep documentation may select regressors that are not compatible (e.g. applying melodic ICA components to the ICA-AROMA output),
or miss regressors that should be used together (e.g. discrete cosine-basis regressors should always be applied with CompCor components).
These inappropriate choices may reintroduce noise to the data or produce suboptimal denoising results.
In addition, there is no standard way of tracking a users' choice of regressors,
and project-specific pipelines can be implemented,
making replication of others denoising steps more difficult. <!-- NC: I thought the two points here could be combined under one issue  -->
Lastly, denoising benchmark literature has thus far been performed on study-specific preprocessing pipelines, 
and so the validity of these results on fMRIPrep has yet to be examined.

## Aim of the benchmark
<!-- NC: I rearranged some sentences here as a different order made sense to me  -->
The current work aims to introduce an application programming interface (API) to standardise user interaction with fMRIPrep and provide a benchmark using functional connectivity generated from resting state data.
We selected two datasets from OpenNeuro,
one with adult and child samples, and the other with psychiatric conditions.
The benchmark will systematically evaluate the impact of common denoising strategies,
their impact on different types of samples,
and select the best approach for a dataset,
providing a useful reference for fMRIPrep users. <!-- NC: I thought some of the sentences here were repetitive so tried to combine them  -->
The code base accompanying this project can be re-executed with different versions of fMRIPrep,
and all material can serve as a foundation to generate connectome based metric quality reports. 
The API is released under popular Python neuroimaging analytic library `nilearn`,
with the aim of maximising the exposure of the API to the larger Python fMRI community.
