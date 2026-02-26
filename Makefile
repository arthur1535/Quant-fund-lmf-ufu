.PHONY: setup test lint format backtest-smoke docker-up docker-down clean

# ============================================================
# QuantNucleo — Makefile de Conveniência
# ============================================================

PYTHON := python
PIP := pip

# --- Setup ---
setup:
	$(PIP) install -e ".[dev,research]"
	pre-commit install
	@echo "✅ Ambiente configurado com sucesso"

# --- Qualidade de Código ---
lint:
	ruff check libs/ services/ strategies/ tests/
	mypy libs/ services/ strategies/

format:
	ruff format libs/ services/ strategies/ tests/

# --- Testes ---
test:
	pytest tests/unit/ -v --tb=short -x

test-all:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=libs --cov=services --cov-report=html --cov-report=term

# --- Backtest ---
backtest-smoke:
	pytest tests/backtest/ -m smoke -v --tb=short

backtest-full:
	pytest tests/backtest/ -v --tb=long

# --- Docker ---
docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down -v

# --- Limpeza ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache .mypy_cache htmlcov

# --- Data Pipeline ---
ingest-raw:
	$(PYTHON) -m libs.data.ingestion.market_data

# --- Serviços ---
api:
	uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000

webhook:
	uvicorn services.webhook_receiver.main:app --reload --host 0.0.0.0 --port 8001
