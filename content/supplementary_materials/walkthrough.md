#  Code documentation

I (HTW) aim to provide some documentation for people who would like to rebuild 
all material of this book.
The code walk through is separated into two parts.
For those who wish to generate all the time series, connectomes, and metric, please see [Setup of this project](setup.md).
If you only want to rebuild the content of the book for the review process, 
or help me edit the manuscript, please see [Rebuild the book](build_book.md).


:::{admonition} I don't want to read the walk through
:class: tip

Here's the TL;DR for building the book.

```{code-block} bash
git clone --recurse-submodules https://github.com/SIMEXP/fmriprep-denoise-benchmark.git
cd fmriprep-denoise-benchmark
virtualenv env 
sourse env/bin/activate
pip install -r binder/requirements.txt
jb build content/
```

:::