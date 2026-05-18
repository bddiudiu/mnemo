.PHONY: install test lint format serve docker clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check mnemo/ tests/
	black --check mnemo/ tests/

format:
	black mnemo/ tests/
	ruff check --fix mnemo/ tests/

serve:
	uvicorn mnemo.api:app --reload --port 8080

docker:
	docker-compose up --build

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
