.PHONY: install test lint format clean

PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin

$(BIN)/python:
	uv venv --python $(PYTHON) $(VENV)

install: $(BIN)/python
	uv pip install --python $(BIN)/python -e ".[dev,llm,api,ui]"

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy src

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
