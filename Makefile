#!/bin/bash

.PHONY: format
format:
	yapf -i -r pipeline_dp tests

.PHONY: lint
lint:
	pylint --rcfile=pylintrc.dms pipeline_dp tests

.PHONY: test
test:
	pytest tests

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -fr

.PHONY: precommit
precommit: lint test format clean

.PHONY: dev
dev: 
	pip install -r requirements.dev.txt
