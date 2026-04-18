.PHONY: dev-setup format check-format lint typecheck test check build release

dev-setup:
	python3 -m pip install -e ".[dev]"

format:
	python3 -m black scripts setup.py trade_assist tests

check-format:
	python3 -m black --check scripts setup.py trade_assist tests

lint:
	python3 -m flake8 trade_assist tests

typecheck:
	python3 -m mypy

test:
	python3 -m pytest -q

check: check-format lint typecheck test

build:
	python3 scripts/release.py --skip-format --skip-tests

release:
	python3 scripts/release.py
