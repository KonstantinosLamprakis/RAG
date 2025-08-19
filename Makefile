.PHONY: format lint check

format:
	black src tests
	isort src tests
	ruff check src tests --fix

lint:
	black --check src tests
	isort --check-only src tests
	ruff check src tests

