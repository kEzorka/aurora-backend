# Точки входа проекта. Фиксируются здесь, чтобы «как это запустить»
# не было устной традицией (docs/README.md).

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.DEFAULT_GOAL := help
.PHONY: help venv setup ingest-analysis ingest-era5 history-monthly forecast validate test serve demo cache-report storage-amplification lint format

help:
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

venv:  ## создать venv и поставить зависимости для тестов
	uv venv --python 3.12 $(VENV)
	$(VENV)/bin/python -m ensurepip --upgrade 2>/dev/null || true
	uv pip install --python $(PY) -r requirements/test-minimal.txt

setup:  ## проверка окружения: ключи, доступность источников, версии, наличие GPU
	$(PY) -m pipeline.setup_check

ingest-analysis:  ## скачать и нормализовать последний доступный анализ
	$(PY) -m pipeline.ingest --kind analysis

ingest-era5:  ## дозалив истории: make ingest-era5 FROM=1990-01-01 TO=1990-12-31
	$(PY) -m pipeline.ingest --kind era5 --from "$(FROM)" --to "$(TO)"

history-monthly:  ## pinned-средние: make history-monthly ROOT=/data/aurora FROM=1940-01-01 TO=2026-04-30T23:00Z
	$(PY) -m pipeline.monthly --root "$(ROOT)" --from "$(FROM)" --to "$(TO)"

forecast:  ## поставить в очередь инференс: make forecast INIT=2026-08-01T00Z
	$(PY) -m pipeline.enqueue --init "$(INIT)"

validate:  ## прогнать валидаторы по последнему записанному срезу
	$(PY) -m validators.cli --latest

test:  ## юнит- и контрактные тесты на фикстурах, без сети
	$(PY) -m pytest

serve:  ## поднять API локально
	$(VENV)/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000

demo:  ## поднять UI + API на локальном синтетическом прогнозе
	$(PY) -m api.demo

cache-report:  ## статистика кэша: занято, hit rate, топ вытеснений
	$(PY) -m cache.report

storage-amplification:  ## таблица логического раздувания чтения для раскладок A/B
	$(PY) -m storage.amplification

lint:  ## ruff + mypy
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .
	$(VENV)/bin/mypy .

format:  ## привести код в порядок
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .
