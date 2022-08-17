# Introductions

<!-- aim of the paragraph: We all know denosing is important now. And we cannot understand brain activity without it. -->
Amongst functional magnetic resonance imaging (fMRI) acquisition protocols,
resting state fMRI has become a tool of choice to explore human brain function in healthy and diseased populations {cite:p}`biswal_2010`.
Functional connectivity {cite:p}`sporns_organization_2004`,
a measure derived from resting state fMRI, is in particular a popular method to characterize the neural basis of <!-- any suggestions of some newer paper to cite is good. -->
cognitive function {cite:p}`finn_functional_2015`,
aging {cite:p}`andrews-hanna_disruption_2007`,
and pathology {cite:p}`woodward_resting-state_2015`.
The rise in popularity of resting state fMRI has also highlighted challenges and gaps in the analytic approach {cite:p}`cole_advances_2010`,
such as distortion introduced by non-neuronal sources.
These include head motion, scanner noise, cardiac and repiratory artifacts {cite:p}`rogers_assessing_2007,murphy_2013`.
Known as confounds or nuisance regressors, they pose a major challenge in functional connectivity analysis {cite:p}`satterthwaite_impact_2012`.
For example, when functional connectivity is contaminated with motion, it can lead to altered network structure {cite:p}`power_scrubbing_2012`.
Indeed, in younger subjects and neurologic/psychiatric patients the impact of head motion on connectivity is frequently reported {cite:p}`makowski_head_2019,satterthwaite_impact_2012`.
Hence, reducing the impact of confounds, known as denoising,
is a major component of the fMRI workflow to ensure the quality of analysis.

## Types of noise regressors

<!-- aim of the paragraph: Classes of nuisance regressors - like how load_confounds separate them -->
The most common method to minimize the impact of confounds in functional connectivity analysis is to perform a linear regression {cite:p}`friston_statistical_1994`.
After basic processing steps (see fMRIPrep {cite:p}`fmriprep1`), 
regressors are regressed out from the signal and the resulting residual used as denoised signal in all subsequent analyses.
Nuisance regressors can be separated into classes, capturing different types of non-neuronal noise.
__Head motion__ is captured by motion realignment measures: _6 rigid-body motion parameters_ (3 translations and 3 rotation) estimated relative to a reference image {cite:p}`friston_movement-related_1996`.
__Non-grey matter tissue signal__ (such as _white matter_ and _cerebrospinal fluid_), unlikely to reflect neuronal activity {cite:p}`fox_pnas_2005`,
is captured by averaging signal within anatomically-derived masks.
__Global signal__ is calculated by averaging signal within the _full brain_ mask {cite:p}`fox_pnas_2005`.
These three classes of regressors can be expanded to their first temporal derivatives and their quadratic terms {cite:p}`satterthwaite_2013` to module non-linear impact of noise. 
Full expansion of head motion parameters is often required for optimal denoising results.
__Scrubbing__ {cite:p}`power_scrubbing_2012` is a volume censoring approach to remove high motion segments in which the framewise displacement 
(see section {ref}`framewise-displacement`)
exceeds some threshold. 
The scrubbing approach is applied along side head motion parameters and tissue signal regressors.
__Temporal high-pass filtering__ accounts for low-frequency signal drifts introduced by _physiological and scanner noise sources_.
Aside from regressors directly modeling noise derived from realignment measures or anatomical properties,
other approaches capture the impact of motion and non-neuronal physiological activity through data-driven methods. 
The principle component based method __CompCor__ {cite:p}`behzadi_compcor_2007,muschelli_compcor_2014` extracts principal components from white matter and cerebrospinal fluid masks to estimate non-neuronal activity.
Independent component analysis based methods estimate spatial independent components representing brain activity and/or noise, 
and then identify the components related to head-motion through using a data-driven classifier (__ICA-FIX__ {cite:p}`salimi-khorshidi_automatic_2014`)
or a pre-trained model (__ICA-AROMA__ {cite:p}`aroma`).

A denoising approach often involves combining a few classes of regressors described above.
Head motion combined with non-grey matter tissue signal is one of the most basic approaches {cite:p}`fox_pnas_2005`. 
Scrubbing combines the basic approach above with volume censoring. 
Anatmoical CompCor regressors are applied along with the basic head motion parameters {cite:p}`muschelli_compcor_2014`.
ICA-AROMA requires further denosing using the basic average of white matter and cerebrospinal fluid signal after the initial independent component denoising {cite:p}`aroma`. 

Different strategies have particular strengths and limitations,
and researchers can make their own choices based on properties of their data and the limitation of the denoising approaches.
One of the major concern is the loss in temporal degrees of freedom, leading to the loss of statistical power.
The more loss in temporal degrees of freedom, the less power the remaining data would have to explain the effects.
The full expansion of head motion, white matter, cerebrospinal fluid, and global signal parameters consists of 36 parameters.
Scrubbing has been shown to mitigate the impact of framewise displacement on the functional connectome, 
but removing volumes prevents analysis focusing on frequency characteristics or dynamic changes in fMRI signal, and reduces the temporal degrees of freedom.
Data-decomposition approaches, such as ICA-AROMA and CompCor, were proposed to preserve statistical power in the data.
These approaches aim to reduce the number of nuisance regressors parameters, <!-- I found this hard to believe (after doing the benchmark), but this statement is in both compcor and the ica paper-->
gain better denoising results without volume censoring,
as well as avoiding potentially overfitting the data and removing meaningful signal {cite:p}`behzadi_compcor_2007,aroma,pruim_evaluation_2015`. 
However, these methods have their own disadvantages.
ICA-AROMA and CompCor can lead to inconsistent number of remaining degrees of freedom as the number of components applied to each subjects can vary.
Past research has also reported that CompCor may only be viable in low-motion data {cite:p}`parkes_evaluation_2018`.

## Implementation of denoising step

<!-- aim of the paragraph: How denoising is traditionally done in propriatory software -->
Which approach to adopt is straightforward when users use preprocessing software with statistical modelling functionality.
Each software implements their own preferred strategy,
for instance, FSL adds the motion parameters by default, and provides an option to run ICA-AROMA.
The noise regressors are preselected,
and thus user intervention is avoided.
However, when users wish to use additional regressors, the process can be complicated.
With the recent benchmark papers on confound denoising highlighting methods such as ICA-AROMA and CompCor {cite:p}`ciric_benchmarking_2017,parkes_evaluation_2018`, 
it's not uncommon for users to want to add parameters other than the native options from a given software.
In the FSL example, if a user wants to use an unsupported strategy, such as CompCor,
they will have to calculate the regressors and inject into the workflow manually.

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
making replication of others denoising steps more difficult.
Lastly, denoising benchmark literature has thus far been performed on study-specific preprocessing pipelines, 
and so the validity of these results on fMRIPrep has yet to be examined.

## Aim of the benchmark

The current work aims to introduce an application programming interface (API) to standardise user interaction with fMRIPrep and provide a benchmark using functional connectivity generated from resting state data.
We selected two datasets on OpenNeuro for the current analysis:
`ds000228` {cite:p}`ds000228:1.1.0` and `ds000030` {cite:p}`ds000030:1.0.0`. 
`ds000228` contains adult and child samples, and `ds000030` includes psychiatric conditions.
The benchmark will systematically evaluate the impact of common denoising strategies,
their impact on different types of samples,
and select the best approach for a dataset,
providing a useful reference for fMRIPrep users. 
The code base accompanying this project can be re-executed with different versions of fMRIPrep,
and all material can serve as a foundation to generate connectome based metric quality reports. 
The API is released under popular Python neuroimaging analytic library `nilearn`,
with the aim of maximising the exposure of the API to the larger Python fMRI community.
