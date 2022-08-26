#  Quick start

```{warning}

The instruction is not working yet as the data on zenodo is not up-to-date yet (26 Aug 2022). 

```

I (HTW) aim to provide some documentation for people who would like to rebuild 
all material of this book.
For those who wish to generate all the time series, connectomes, and metric, 
please see [Setup to run all customised scripts for preprocessing](setup.md).

If you only want to rebuild the content of the book for the review process, 
or help me edit the manuscript, the instruction is as followed:

:::{admonition} Here's the TL;DR for building the book.
:class: tip

```{code-block} bash
git clone --recurse-submodules https://github.com/SIMEXP/fmriprep-denoise-benchmark.git
cd fmriprep-denoise-benchmark
virtualenv env 
sourse env/bin/activate
pip install -r binder/requirements.txt
make all
```
:::

