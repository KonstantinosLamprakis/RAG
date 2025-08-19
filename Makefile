.PHONY: lint lint-fix test docker-prune docker-delete

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

docker-prune:
	docker system prune -f
	docker volume prune -f
	docker image prune -f
	docker network prune -f

docker-delete: docker-prune
	docker volume rm $$(docker volume ls -q) 2>/dev/null || true