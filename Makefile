.PHONY: help atlas templateflow book all

PYTHON ?= python

all: atlas templateflow book

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  atlas				to download atlases for analysis"
	@echo "  templateflow		to process the atlases for analysis"
	@echo "  data				to download data needed for jupyter book"
	@echo "  book				to compile the jupyter book"
	@echo "  all				to run 'atlas', templateflow, and 'book'"

atlas:
	@echo "Download the original atlases..."
	$(PYTHON) scripts/fetch_templates.py

templateflow:
	@echo "Prepare for Templateflow..."
	bash scripts/setup_templateflow.py

book:
	jb build content --all