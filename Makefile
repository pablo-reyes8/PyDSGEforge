.DEFAULT_GOAL := help

PYTHON ?= python3
PIP := $(PYTHON) -m pip

.PHONY: help install install-dev test test-cov lint format check build clean \
	docker-build docker-test showcase dynare-check

help: ## Show the available development commands.
	@awk 'BEGIN {FS = ":.*## "; printf "PyDSGEforge development commands:\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the library in editable mode.
	$(PIP) install -e .

install-dev: ## Install the library and development tools.
	$(PIP) install -e ".[dev]"

test: ## Run the test suite.
	$(PYTHON) -m pytest -q

test-cov: ## Run tests with branch coverage.
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing --cov-report=xml

lint: ## Check correctness-critical lint rules with Ruff.
	$(PYTHON) -m ruff check src tests scripts dynare_comprobation/validate.py

format: ## Format Python files and fix safe lint findings.
	$(PYTHON) -m ruff format src tests scripts dynare_comprobation/validate.py
	$(PYTHON) -m ruff check --fix src tests scripts dynare_comprobation/validate.py

check: lint test dynare-check build ## Run all local quality gates.

build: ## Build source and wheel distributions.
	$(PYTHON) -m build

clean: ## Remove local build and test artifacts.
	rm -rf build dist htmlcov .pytest_cache .ruff_cache .coverage coverage.xml *.egg-info

docker-build: ## Build the minimal runtime image.
	docker build --pull --target runtime -t pydsgeforge:local .

docker-test: ## Build and run the isolated test image.
	docker build --target test -t pydsgeforge:test .
	docker run --rm pydsgeforge:test

showcase: ## Rebuild showcase datasets, estimates, and figures.
	$(PYTHON) scripts/build_showcase.py

dynare-check: ## Re-evaluate the committed Dynare parity fixture.
	$(PYTHON) dynare_comprobation/validate.py
