# Introductions

Selecting an optimal denoising strategy is a key issue when processing fMRI data. 
<!-- Popular fMRI data processing software often contain preprocessing and later statistical modelling in one package.
Since each step is linked to each other, the generated motion confounds were all designed to use as is.
However, with the recent benchmark papers on confound denoising
 -->
The popular software fMRIPrep {cite:p}`esteban_fmriprep_2020` aims to standardize fMRI preprocessing, 
but users are still offered a wide range of confound regressors to choose from to denoise data. 
Without a good understanding of the literature or the fMRIPrep documentation, 
users can select suboptimal strategies. 
Current work aims to provide a useful reference for fMRIPrep users by systematically evaluating the impact of different confound regression strategies, 
and by contrasting the results with past literature based on alternative preprocessing software.  
