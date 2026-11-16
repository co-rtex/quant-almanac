PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy
PYTEST ?= .venv/bin/pytest
UVICORN ?= .venv/bin/uvicorn
ALEMBIC ?= .venv/bin/alembic

.PHONY: install lint typecheck test e2e validate migrate run-api db-up db-down demo demo-console install-e2e

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

install-e2e:
	$(PIP) install -e .[dev,e2e]
	.venv/bin/playwright install chromium

lint:
	$(RUFF) check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

e2e:
	$(PYTHON) -m pytest tests/e2e -m e2e

validate: lint typecheck test

migrate:
	$(ALEMBIC) upgrade head

run-api:
	$(UVICORN) quant_signal.api.app:app --reload --host 127.0.0.1 --port 8000

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

demo:
	.venv/bin/quant-almanac-demo run

demo-console:
	.venv/bin/quant-almanac-demo serve
