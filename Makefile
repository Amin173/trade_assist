.PHONY: dev-setup format check-format lint typecheck test check build release

STYLE_TARGETS = scripts setup.py trade_assist tests
AUTOFLAKE_FLAGS = --remove-all-unused-imports --remove-unused-variables

dev-setup:
	python3 -m pip install -e ".[dev]"

format:
	python3 -m autoflake --in-place --recursive $(AUTOFLAKE_FLAGS) $(STYLE_TARGETS)
	python3 -m black $(STYLE_TARGETS)

check-format:
	python3 -m autoflake --check-diff --recursive $(AUTOFLAKE_FLAGS) $(STYLE_TARGETS)
	python3 -m black --check $(STYLE_TARGETS)

lint:
	python3 -m flake8 $(STYLE_TARGETS)

typecheck:
	python3 -m mypy

test:
	python3 -m pytest -q

check: check-format lint typecheck test

build:
	python3 scripts/release.py --skip-format --skip-lint --skip-typecheck --skip-tests

release:
	python3 scripts/release.py
