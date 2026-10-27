.PHONY: install test lint run docker-build docker-up clean generate-data train

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -v --tb=short

lint:
	flake8 src/ --max-line-length=100 --ignore=E501,W503

run:
	PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

generate-data:
	PYTHONPATH=. python -m src.data.synthetic_generator

train:
	PYTHONPATH=. python -c "from src.data.synthetic_generator import generate_synthetic_data; print('Data generated')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov
