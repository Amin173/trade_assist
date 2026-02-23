.PHONY: format check-format lint typecheck test check

format:
	python3 -m black trade_assist tests

check-format:
	python3 -m black --check trade_assist tests

lint:
	python3 -m flake8 trade_assist tests

typecheck:
	python3 -m mypy

test:
	python3 -m pytest -q

check: check-format lint typecheck test
