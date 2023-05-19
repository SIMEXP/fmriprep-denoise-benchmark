.PHONY: help data book all

PYTHON ?= python

all: data book figures

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  data				to download data needed for jupyter book with repo2data"
	@echo "  book				to compile the jupyter book"
	@echo "  figures			to make all manuscript figures"
	@echo "  all				to run 'atlas', templateflow, and 'book'"

data:
	@echo "Download input data to build the report"
	cd content/notebooks &&	repo2data -r ../../binder/data_requirement.json && cd ../..

book:
	jb build content --all

figures: data/fmriprep-denoise-benchmark/denoise-metrics
	$(PYTHON) scripts/make_manuscript_figures.py
