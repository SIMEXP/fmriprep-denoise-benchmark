.PHONY: help atlas data book all

all: atlas data book

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  atlas				to download atlases for analysis"
	@echo "  data				to download data needed for jupyter book"
	@echo "  book				to compile the jupyter book"
	@echo "  all				to run 'atlas', templateflow, and 'book'"

atlas:
	@echo "Download the original atlases..."
	if [ ! -d inputs/custome_templateflow ]; then wget -c -O custome_templateflow.tar.gz "https://zenodo.org/record/7362211/files/custome_templateflow.tar.gz?download=1" && mkdir -p inputs && tar xf custome_templateflow.tar.gz -C inputs && rm custome_templateflow.tar.gz; fi

data:
	@echo "Download input data to build the report"
	if [ ! -d inputs/denoise-metrics ]; then wget -c -O denoise-metrics.tar.gz "https://zenodo.org/record/7362211/files/denoise-metrics.tar.gz?download=1" && mkdir -p inputs && tar xf denoise-metrics.tar.gz -C inputs && rm denoise-metrics.tar.gz; fi

book:
	jb build content --all