install: FORCE
	pip install -e .[test]

lint: FORCE
	flake8
	black --check .
	isort --check .
	mypy .

format: FORCE
	black .
	isort .

test: lint FORCE
	python example.py
	# pytest -v -n auto test

FORCE:
