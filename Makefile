.PHONY: dev test lint train compose-up

dev:
	cd backend && uvicorn app.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check backend

train:
	cd backend && python -m app.ml.train

compose-up:
	docker compose up --build
