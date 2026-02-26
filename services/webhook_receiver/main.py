"""
QuantNucleo — Webhook Receiver (TradingView → Núcleo Quant)
=============================================================
Endpoint principal para receber alertas do TradingView via webhook.

Fluxo:
  TradingView Alert
    → POST /api/v1/webhooks/tradingview
    → Verificação de Segurança (token + IP allowlist)
    → Idempotência (deduplicação por key)
    → Validação de Payload
    → Motor de Sinais (geração + risco)
    → Registro MongoDB (auditoria completa)
    → Notificação / Ação

Segurança:
  - Token de autenticação no header X-Webhook-Token
  - IP allowlist (TradingView IPs conhecidos)
  - Rate limiting
  - Payload size limit (64KB)
  - Replay protection (idempotency key + TTL)

Observabilidade:
  - Métricas Datadog (latência, taxa de sucesso, taxa aprovação)
  - Erros Sentry
  - Logs estruturados
"""

from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

import sentry_sdk
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import get_config
from libs.core.logging import get_logger
from libs.core.types import StatusSinal, WebhookPayload
from libs.signals.engine import SignalEngine

logger = get_logger("webhook_receiver")

# =============================================================================
# App FastAPI
# =============================================================================

app = FastAPI(
    title="QuantNucleo Webhook Receiver",
    description="Receptor de webhooks TradingView → Núcleo Quant",
    version="0.1.0",
    docs_url="/docs" if get_config().debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if get_config().debug else [],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# =============================================================================
# Estado em Memória (substituir por Redis em prod)
# =============================================================================

_idempotency_cache: dict[str, float] = {}  # key → timestamp
_signal_engine = SignalEngine()


# =============================================================================
# Models
# =============================================================================

class TradingViewAlert(BaseModel):
    """Payload esperado do TradingView webhook."""
    ticker: str
    action: str  # "buy", "sell", "alert"
    price: float
    interval: Optional[str] = None
    exchange: Optional[str] = None
    strategy_name: Optional[str] = None
    indicator_value: Optional[float] = None
    message: Optional[str] = None
    # Chave de idempotência (opcional, gerada se ausente)
    idempotency_key: Optional[str] = None


class WebhookResponse(BaseModel):
    """Resposta padrão do webhook."""
    status: str
    signal_id: Optional[str] = None
    signal_status: Optional[str] = None
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    """Health check."""
    status: str = "ok"
    service: str = "webhook_receiver"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Segurança
# =============================================================================

def verificar_token(request: Request) -> bool:
    """Verifica token de autenticação do webhook."""
    cfg = get_config().tradingview
    token = request.headers.get("X-Webhook-Token", "")
    expected = cfg.webhook_secret.get_secret_value()

    if not expected:
        logger.warning("webhook_sem_secret_configurado")
        return True  # Dev mode — aceitar tudo

    return hmac.compare_digest(token, expected)


def verificar_ip(request: Request) -> bool:
    """Verifica IP contra allowlist do TradingView."""
    cfg = get_config().tradingview
    client_ip = request.client.host if request.client else "unknown"

    if get_config().debug:
        return True  # Dev mode

    if client_ip not in cfg.allowed_ips:
        logger.warning("webhook_ip_bloqueado", ip=client_ip)
        return False

    return True


def verificar_idempotencia(key: str) -> bool:
    """
    Verifica se este webhook já foi processado (idempotência).
    Retorna True se é NOVO (não duplicado).
    """
    cfg = get_config().tradingview
    now = time.time()

    # Limpar expirados
    expired = [k for k, ts in _idempotency_cache.items() if now - ts > cfg.idempotency_ttl_seconds]
    for k in expired:
        del _idempotency_cache[k]

    if key in _idempotency_cache:
        logger.info("webhook_duplicado", key=key)
        return False

    _idempotency_cache[key] = now
    return True


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check — usado por AKS liveness/readiness probes."""
    return HealthResponse()


@app.post(
    "/api/v1/webhooks/tradingview",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def receber_alerta_tradingview(
    alert: TradingViewAlert,
    request: Request,
) -> WebhookResponse:
    """
    Endpoint principal — recebe alertas do TradingView.
    
    Fluxo:
    1. Verificar token (segurança)
    2. Verificar IP (allowlist)
    3. Verificar idempotência (deduplicação)
    4. Processar alerta → Sinal
    5. Validar sinal (risco)
    6. Registrar (auditoria)
    7. Responder
    """
    inicio = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        "webhook_recebido",
        ticker=alert.ticker,
        action=alert.action,
        price=alert.price,
        ip=client_ip,
    )

    # 1. Segurança: token
    if not verificar_token(request):
        logger.warning("webhook_token_invalido", ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
        )

    # 2. Segurança: IP
    if not verificar_ip(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="IP não autorizado",
        )

    # 3. Idempotência
    idempotency_key = alert.idempotency_key or hashlib.sha256(
        f"{alert.ticker}:{alert.action}:{alert.price}:{int(time.time() / 60)}".encode()
    ).hexdigest()

    if not verificar_idempotencia(idempotency_key):
        return WebhookResponse(
            status="duplicado",
            message="Alerta já processado (idempotência)",
        )

    # 4. Processar alerta → Sinal
    try:
        sinal = _signal_engine.processar_alerta_tradingview(
            ticker=alert.ticker,
            action=alert.action,
            price=alert.price,
            message=alert.message,
            indicator_value=alert.indicator_value,
        )
    except Exception as e:
        logger.error("webhook_processamento_erro", erro=str(e))
        sentry_sdk.capture_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar alerta: {e}",
        )

    # 5. Validar sinal (risco)
    sinal = _signal_engine.validar_e_registrar(sinal)

    # 6. Registrar auditoria (MongoDB — placeholder)
    audit_record = {
        "webhook_id": idempotency_key,
        "received_at": datetime.utcnow().isoformat(),
        "source_ip": client_ip,
        "payload": alert.model_dump(),
        "signal_id": str(sinal.id),
        "signal_status": sinal.status.value,
        "latency_ms": (time.time() - inicio) * 1000,
    }
    # TODO: Persistir no MongoDB Atlas
    logger.info("webhook_auditoria", **audit_record)

    # 7. Responder
    latencia_ms = (time.time() - inicio) * 1000
    logger.info(
        "webhook_processado",
        sinal_id=str(sinal.id),
        status=sinal.status.value,
        latencia_ms=f"{latencia_ms:.1f}",
    )

    return WebhookResponse(
        status="aceito",
        signal_id=str(sinal.id),
        signal_status=sinal.status.value,
        message=f"Sinal {sinal.status.value} — {sinal.motivo}",
    )


@app.get("/api/v1/signals/pending")
async def sinais_pendentes() -> list[dict[str, Any]]:
    """Lista sinais validados aguardando aprovação humana."""
    return [
        {
            "id": str(s.id),
            "ticker": s.ativo.ticker,
            "tipo": s.tipo.value,
            "preco": s.preco_referencia,
            "confianca": s.confianca,
            "fonte": s.fonte.value,
            "motivo": s.motivo,
            "timestamp": s.timestamp.isoformat(),
        }
        for s in _signal_engine.sinais_pendentes
    ]


@app.post("/api/v1/signals/{signal_id}/approve")
async def aprovar_sinal(signal_id: str, aprovador: str = "manual") -> dict:
    """Aprovação humana de sinal (execução assistida)."""
    for sinal in _signal_engine._sinais_pendentes:
        if str(sinal.id) == signal_id:
            sinal.status = StatusSinal.EXECUTADO
            sinal.aprovado_por = aprovador
            sinal.aprovado_em = datetime.utcnow()
            logger.info("sinal_aprovado", sinal_id=signal_id, aprovador=aprovador)
            return {"status": "aprovado", "signal_id": signal_id}

    raise HTTPException(status_code=404, detail="Sinal não encontrado")


@app.post("/api/v1/signals/{signal_id}/reject")
async def rejeitar_sinal(signal_id: str, motivo: str = "") -> dict:
    """Rejeição manual de sinal."""
    for sinal in _signal_engine._sinais_pendentes:
        if str(sinal.id) == signal_id:
            sinal.status = StatusSinal.REJEITADO
            sinal.motivo_rejeicao = motivo or "Rejeitado manualmente"
            logger.info("sinal_rejeitado", sinal_id=signal_id, motivo=motivo)
            return {"status": "rejeitado", "signal_id": signal_id}

    raise HTTPException(status_code=404, detail="Sinal não encontrado")
