# Makefile for Python project
#
# The build will automatically fail, if at least one check fails.
.PHONY: black
black:
	black --check dualnumber/

.PHONY: lint
lint:
	pylint --rcfile pylintrc dualnumber/

.PHONY: typehint
typehint:
	mypy dualnumber/

.PHONY: test
test:
	pytest tests/

.PHONY: check 
check: black lint typehint test

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -rf
	find . -type d -name __pycache__ | xargs rm -rf