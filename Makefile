.PHONY: lint lint-fix test

lint-fix:
	black src tests
	isort src tests
	ruff check src tests --fix

lint:
	black --check src tests
	isort --check-only src tests
	ruff check src tests

test:
	pytest --maxfail=1 --disable-warnings -q