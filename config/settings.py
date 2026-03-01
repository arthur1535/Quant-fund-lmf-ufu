# QuantNucleo — Configuração Central
# ====================================
# Este arquivo centraliza TODA configuração do núcleo quant.
# Variáveis sensíveis vêm do Azure Key Vault / .env (NUNCA hardcoded).

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Ambiente(str, Enum):
    """Ambientes de execução."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class AzureConfig(BaseSettings):
    """Configuração Azure — populada via env vars ou Key Vault."""
    model_config = SettingsConfigDict(env_prefix="AZURE_")

    subscription_id: str = ""
    resource_group: str = "rg-quantnucleo"
    storage_account: str = "stquantnucleo"
    storage_container_raw: str = "raw"
    storage_container_processed: str = "processed"
    storage_container_features: str = "features"
    keyvault_url: str = ""
    synapse_workspace: str = ""
    synapse_pool: str = "quantpool"


class MongoConfig(BaseSettings):
    """MongoDB Atlas — logs operacionais, auditoria, telemetria."""
    model_config = SettingsConfigDict(env_prefix="MONGO_")

    uri: SecretStr = SecretStr("")
    database: str = "quantnucleo"
    collection_logs: str = "execution_logs"
    collection_audit: str = "webhook_audit"
    collection_signals: str = "signals"
    collection_telemetry: str = "telemetry"


class TradingViewConfig(BaseSettings):
    """TradingView — webhooks e alertas."""
    model_config = SettingsConfigDict(env_prefix="TV_")

    webhook_secret: SecretStr = SecretStr("")
    webhook_endpoint: str = "/api/v1/webhooks/tradingview"
    allowed_ips: list[str] = Field(default_factory=lambda: [
        "52.89.214.238", "34.212.75.30",
        "54.218.53.128", "52.32.178.7"
    ])
    idempotency_ttl_seconds: int = 3600
    max_payload_size_bytes: int = 65536


class DatadogConfig(BaseSettings):
    """Datadog — observabilidade e métricas quant."""
    model_config = SettingsConfigDict(env_prefix="DD_")

    api_key: SecretStr = SecretStr("")
    app_key: SecretStr = SecretStr("")
    service: str = "quantnucleo"
    env: str = "dev"
    enable_tracing: bool = True


class SentryConfig(BaseSettings):
    """Sentry — tracking de erros runtime."""
    model_config = SettingsConfigDict(env_prefix="SENTRY_")

    dsn: SecretStr = SecretStr("")
    environment: str = "dev"
    traces_sample_rate: float = 0.1
    profiles_sample_rate: float = 0.1


class MarketDataConfig(BaseSettings):
    """Configuração de fontes de dados de mercado."""
    model_config = SettingsConfigDict(env_prefix="MARKET_")

    # yfinance
    default_interval: str = "1d"
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    
    # BCB API (dados brasileiros)
    bcb_api_base: str = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
    
    # CVM API (fundos brasileiros)
    cvm_api_base: str = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS"
    
    # Fallback
    fallback_cache_hours: int = 24


class RiskConfig(BaseSettings):
    """Parâmetros do motor de risco."""
    model_config = SettingsConfigDict(env_prefix="RISK_")

    # VaR / CVaR
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    
    # Monte Carlo
    mc_simulations: int = 10_000
    mc_horizon_days: int = 252
    
    # Drawdown
    max_drawdown_threshold: float = -0.15  # -15%
    
    # Position Limits
    max_single_position: float = 0.30  # 30% máx por ativo
    max_sector_exposure: float = 0.50  # 50% máx por setor
    
    # Trading days/year
    trading_days_year: int = 252
    
    # GARCH
    garch_p: int = 1
    garch_q: int = 1
    garch_dist: str = "normal"


class LSTMConfig(BaseSettings):
    """Parâmetros padrão LSTM."""
    model_config = SettingsConfigDict(env_prefix="LSTM_")

    hidden_units: int = 50
    dropout: float = 0.2
    look_back: int = 60
    forecast_horizon: int = 30
    epochs: int = 30
    batch_size: int = 32
    patience: int = 5
    seed: int = 42


class QuantNucleoConfig(BaseSettings):
    """Configuração raiz — agrega todas as sub-configurações."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ambiente: Ambiente = Ambiente.DEV
    project_name: str = "quantnucleo"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Sub-configurações
    azure: AzureConfig = Field(default_factory=AzureConfig)
    mongo: MongoConfig = Field(default_factory=MongoConfig)
    tradingview: TradingViewConfig = Field(default_factory=TradingViewConfig)
    datadog: DatadogConfig = Field(default_factory=DatadogConfig)
    sentry: SentryConfig = Field(default_factory=SentryConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)

    @property
    def is_prod(self) -> bool:
        return self.ambiente == Ambiente.PROD


import functools


@functools.lru_cache(maxsize=1)
def get_config() -> QuantNucleoConfig:
    """Factory — retorna configuração singleton (cached)."""
    return QuantNucleoConfig()
