install: FORCE
	python -m pip install -e .[test]

lint: FORCE
	python -m flake8
	python -m black --check *.py pyroed test examples/*.py
	python -m isort --check .
	python -m mypy --install-types --non-interactive pyroed

format: FORCE
	python -m black *.py pyroed test examples/*.py
	python -m isort .

test: lint FORCE
	pytest -vx test
	python examples/immune_sequence.py --simulate-experiments=1
	@echo PASSED

FORCE:
