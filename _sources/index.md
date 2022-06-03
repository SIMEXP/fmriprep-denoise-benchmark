# Overview

The project is a continuation of [`load_confounds`](https://github.com/SIMEXP/load_confounds). 
The aim is to evaluate the impact of denoising strategy on functional connectivity data, using output processed by fMRIPrep LTS.

The main tool is now part of 
[`nilearn`](https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html#nilearn.interfaces.fmriprep.load_confounds).

The preliminary results will be presented at OHBM 2022 as a poster. 
Please find poster number `WTh570`.

```{image} ./images/ohbm2022_abstract_head.png
:alt: poster
:align: center
```

## Acknowledgments

The initial API was started by Hanad Sharmarke[^1] and Pierre Bellec[^1][^7].
The implementation was completed by Hao-Ting Wang[^1], Steven Meisler[^2][^3], François Paugam[^1][^4], Pierre Bellec.
Hao-Ting Wang migrated the code base to `nilearn`.
Nicolas Gensollen[^5] and Bertrand Thirion[^5] reviewed the code migrated to `nilearn`.
We thank Chris Markiewicz[^6] for feedbacks related to fMRIPrep.

Hao-Ting Wang and Pierre Bellec drafted the paper.

Please see the [original repository](https://github.com/SIMEXP/load_confounds#contributors-) for a full history of development and contributors.

[^1]: Centre de recherche de l'institut Universitaire de gériatrie de Montréal (CRIUGM), Montréal, Québec, Canada

[^2]: Harvard University, MA, USA

[^3]: Massachusetts Institute of Technology, MA, USA

[^4]: Computer Science and Operations Research Department, Université de Montréal, Montréal, Québec, Canada

[^5]: Inria, CEA, Université Paris-Saclay, Paris, France

[^6]: Department of Psychology, Stanford University, Stanford, United States

[^7]: Psychology Department, Université de Montréal, Montréal, Québec, Canada