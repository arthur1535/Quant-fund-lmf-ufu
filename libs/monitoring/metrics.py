# =====================================================================
# QuantNucleo — Configuração de Monitoramento e Observabilidade
# =====================================================================
# Integrações: Datadog, Sentry, Application Insights
# Métricas customizadas para operações quantitativas

from __future__ import annotations

import time
import functools
from typing import Any, Callable
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


# =====================================================================
# Métricas Quantitativas
# =====================================================================

@dataclass
class MetricaQuant:
    """Definição de uma métrica customizada para monitoramento."""
    nome: str
    tipo: str  # gauge, counter, histogram
    descricao: str
    tags: list[str] = field(default_factory=list)
    unidade: str = ""
    alerta_threshold: float | None = None


# Catálogo de métricas do QuantNucleo
METRICAS_CATALOGO: list[MetricaQuant] = [
    # --- Risco ---
    MetricaQuant(
        nome="quantnucleo.risk.var_95",
        tipo="gauge",
        descricao="Value at Risk 95% corrente do portfólio",
        tags=["portfolio", "risk"],
        unidade="percent",
        alerta_threshold=-5.0,
    ),
    MetricaQuant(
        nome="quantnucleo.risk.cvar_95",
        tipo="gauge",
        descricao="Conditional VaR 95%",
        tags=["portfolio", "risk"],
        unidade="percent",
    ),
    MetricaQuant(
        nome="quantnucleo.risk.drawdown_max",
        tipo="gauge",
        descricao="Drawdown máximo atual",
        tags=["portfolio", "risk"],
        unidade="percent",
        alerta_threshold=-15.0,
    ),
    MetricaQuant(
        nome="quantnucleo.risk.sharpe_rolling",
        tipo="gauge",
        descricao="Sharpe Ratio rolling 60d",
        tags=["portfolio", "performance"],
    ),
    MetricaQuant(
        nome="quantnucleo.risk.volatilidade_garch",
        tipo="gauge",
        descricao="Volatilidade forecasted via GARCH(1,1)",
        tags=["portfolio", "risk", "garch"],
        unidade="percent",
    ),

    # --- Sinais ---
    MetricaQuant(
        nome="quantnucleo.signals.recebidos",
        tipo="counter",
        descricao="Total de sinais recebidos (todas as fontes)",
        tags=["signals"],
    ),
    MetricaQuant(
        nome="quantnucleo.signals.aprovados",
        tipo="counter",
        descricao="Sinais aprovados e executados",
        tags=["signals"],
    ),
    MetricaQuant(
        nome="quantnucleo.signals.rejeitados",
        tipo="counter",
        descricao="Sinais rejeitados pelo risk engine",
        tags=["signals", "risk"],
    ),
    MetricaQuant(
        nome="quantnucleo.signals.pendentes",
        tipo="gauge",
        descricao="Sinais aguardando aprovação humana",
        tags=["signals"],
        alerta_threshold=10.0,
    ),

    # --- Webhook ---
    MetricaQuant(
        nome="quantnucleo.webhook.latencia",
        tipo="histogram",
        descricao="Latência de processamento de webhooks (ms)",
        tags=["webhook", "latency"],
        unidade="millisecond",
        alerta_threshold=500.0,
    ),
    MetricaQuant(
        nome="quantnucleo.webhook.recebidos",
        tipo="counter",
        descricao="Total de webhooks recebidos",
        tags=["webhook"],
    ),
    MetricaQuant(
        nome="quantnucleo.webhook.duplicados",
        tipo="counter",
        descricao="Webhooks duplicados (idempotency hit)",
        tags=["webhook"],
    ),
    MetricaQuant(
        nome="quantnucleo.webhook.bloqueados",
        tipo="counter",
        descricao="Webhooks bloqueados (IP/auth)",
        tags=["webhook", "security"],
    ),

    # --- LSTM ---
    MetricaQuant(
        nome="quantnucleo.lstm.mse_treino",
        tipo="gauge",
        descricao="MSE do último treino LSTM",
        tags=["lstm", "ml"],
    ),
    MetricaQuant(
        nome="quantnucleo.lstm.previsao_desvio",
        tipo="gauge",
        descricao="Desvio médio das previsões vs realizado (%)",
        tags=["lstm", "ml"],
        unidade="percent",
    ),

    # --- Infraestrutura ---
    MetricaQuant(
        nome="quantnucleo.infra.heartbeat",
        tipo="gauge",
        descricao="Heartbeat do sistema (1 = OK, 0 = DOWN)",
        tags=["infra"],
        alerta_threshold=0.5,
    ),
    MetricaQuant(
        nome="quantnucleo.infra.api_latencia_p99",
        tipo="histogram",
        descricao="Latência P99 da API principal",
        tags=["infra", "api"],
        unidade="millisecond",
        alerta_threshold=1000.0,
    ),
]


# =====================================================================
# Coletor de Métricas (abstração sobre Datadog/StatsD)
# =====================================================================

class MetricsCollector:
    """
    Coletor central de métricas.
    Em produção, envia para Datadog via DogStatsD.
    Em dev, apenas loga.
    """

    def __init__(self, ambiente: str = "dev"):
        self.ambiente = ambiente
        self._client = None
        self._inicializar()

    def _inicializar(self):
        if self.ambiente == "prod":
            try:
                from datadog import initialize, statsd
                initialize(statsd_host="localhost", statsd_port=8125)
                self._client = statsd
                logger.info("📊 Datadog StatsD inicializado")
            except ImportError:
                logger.warning("⚠️ datadog não instalado, usando log fallback")
        else:
            logger.info("📊 Métricas em modo log (dev)")

    def gauge(self, nome: str, valor: float, tags: list[str] | None = None):
        tags = tags or []
        tags.append(f"env:{self.ambiente}")
        if self._client:
            self._client.gauge(nome, valor, tags=tags)
        else:
            logger.debug("metric.gauge", metric=nome, value=valor, tags=tags)

    def increment(self, nome: str, valor: int = 1, tags: list[str] | None = None):
        tags = tags or []
        tags.append(f"env:{self.ambiente}")
        if self._client:
            self._client.increment(nome, valor, tags=tags)
        else:
            logger.debug("metric.counter", metric=nome, value=valor, tags=tags)

    def histogram(self, nome: str, valor: float, tags: list[str] | None = None):
        tags = tags or []
        tags.append(f"env:{self.ambiente}")
        if self._client:
            self._client.histogram(nome, valor, tags=tags)
        else:
            logger.debug("metric.histogram", metric=nome, value=valor, tags=tags)

    def timing(self, nome: str):
        """Decorator para medir tempo de execução."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                inicio = time.perf_counter()
                resultado = func(*args, **kwargs)
                duracao_ms = (time.perf_counter() - inicio) * 1000
                self.histogram(nome, duracao_ms)
                return resultado
            return wrapper
        return decorator


# =====================================================================
# Alertas e Monitors (definições Datadog)
# =====================================================================

DATADOG_MONITORS: list[dict[str, Any]] = [
    {
        "name": "[QuantNucleo] VaR 95% acima do limite",
        "type": "metric alert",
        "query": "avg(last_15m):avg:quantnucleo.risk.var_95{env:prod} < -5",
        "message": "🚨 VaR 95% do portfólio abaixo de -5%. Risco elevado! @pagerduty-quantnucleo",
        "priority": 1,
        "tags": ["team:quant", "service:risk-engine"],
    },
    {
        "name": "[QuantNucleo] Drawdown máximo excedido",
        "type": "metric alert",
        "query": "avg(last_30m):avg:quantnucleo.risk.drawdown_max{env:prod} < -15",
        "message": "🔴 Drawdown > 15%. Possível circuit breaker. @slack-quantnucleo-alerts",
        "priority": 1,
        "tags": ["team:quant", "service:risk-engine"],
    },
    {
        "name": "[QuantNucleo] Webhook latência alta",
        "type": "metric alert",
        "query": "avg(last_5m):p99:quantnucleo.webhook.latencia{env:prod} > 500",
        "message": "⚠️ Latência P99 de webhooks > 500ms. Verificar processamento. @slack-quantnucleo-infra",
        "priority": 2,
        "tags": ["team:quant", "service:webhook"],
    },
    {
        "name": "[QuantNucleo] Sinais pendentes acumulando",
        "type": "metric alert",
        "query": "avg(last_15m):avg:quantnucleo.signals.pendentes{env:prod} > 10",
        "message": "⚠️ Mais de 10 sinais pendentes. Aprovação manual necessária. @slack-quantnucleo-trading",
        "priority": 2,
        "tags": ["team:quant", "service:signals"],
    },
    {
        "name": "[QuantNucleo] Heartbeat DOWN",
        "type": "metric alert",
        "query": "avg(last_5m):avg:quantnucleo.infra.heartbeat{env:prod} < 1",
        "message": "🚨 Sistema fora do ar! Verificação imediata. @pagerduty-quantnucleo",
        "priority": 1,
        "tags": ["team:quant", "service:infra"],
    },
]
