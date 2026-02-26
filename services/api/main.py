"""
QuantNucleo — API REST Principal
==================================
API central do núcleo quant para:
 - Status do portfolio
 - Métricas de risco
 - Histórico de sinais
 - Interface para backtesting
 - Health checks
"""

from __future__ import annotations

from datetime import datetime

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import get_config
from libs.core.logging import get_logger

logger = get_logger("api")
cfg = get_config()

# Inicializar Sentry
if cfg.sentry.dsn.get_secret_value():
    sentry_sdk.init(
        dsn=cfg.sentry.dsn.get_secret_value(),
        environment=cfg.sentry.environment,
        traces_sample_rate=cfg.sentry.traces_sample_rate,
        profiles_sample_rate=cfg.sentry.profiles_sample_rate,
    )

app = FastAPI(
    title="QuantNucleo API",
    description="API Central — Núcleo Quantitativo LMF/UFU",
    version=cfg.version,
    docs_url="/docs" if cfg.debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cfg.debug else [],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "quantnucleo-api"
    version: str = cfg.version
    ambiente: str = cfg.ambiente.value
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/v1/status")
async def status_nucleo() -> dict:
    """Status geral do núcleo quant."""
    return {
        "status": "operacional",
        "ambiente": cfg.ambiente.value,
        "version": cfg.version,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "ok",
            "webhook_receiver": "ok",
            "signal_engine": "ok",
            "risk_gateway": "ok",
        },
    }


@app.get("/api/v1/risk/summary")
async def risk_summary() -> dict:
    """Resumo de risco atual (placeholder)."""
    return {
        "portfolio_var_95": -0.023,
        "portfolio_cvar_95": -0.035,
        "max_drawdown_30d": -0.08,
        "sharpe_30d": 1.45,
        "sortino_30d": 1.82,
        "sinais_pendentes": 0,
        "alertas_nao_validados": 0,
        "timestamp": datetime.utcnow().isoformat(),
        "nota": "Dados placeholder — conectar ao motor de risco em produção",
    }


@app.get("/api/v1/signals/history")
async def signals_history(limit: int = 50) -> dict:
    """Histórico de sinais gerados."""
    return {
        "total": 0,
        "sinais": [],
        "nota": "Conectar ao MongoDB Atlas para histórico persistente",
    }
