#!/bin/bash

.PHONY: format
format:
	poetry run yapf -i -r pipeline_dp analysis tests examples || \
	    yapf3 -i -r pipeline_dp analysis tests examples || \
	    python3 -m yapf -i -r pipeline_dp analysis tests examples

.PHONY: lint
lint:
	poetry run pylint --rcfile=pylintrc.dms pipeline_dp tests

.PHONY: test
test:
	poetry run pytest tests analysis/tests

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -fr

.PHONY: precommit
precommit: lint test format clean

.PHONY: dev
dev: 
	poetry install

.PHONY: dist
dist: 
	python setup.py bdist_wheel