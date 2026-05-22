CONDA ?= conda
CONDA_ENV ?= mnist-nn
PYTHON ?= $(CONDA) run -n base python

.PHONY: setup test download

setup:
	$(PYTHON) scripts/setup_env.py --conda "$(CONDA)" --env "$(CONDA_ENV)" --file environment.yml

test:
	$(CONDA) run -n "$(CONDA_ENV)" python -m pytest tests -v

download:
	$(CONDA) run -n "$(CONDA_ENV)" python download_mnist.py
