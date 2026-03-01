"""
QuantNucleo — Tipos e modelos de domínio
=========================================
Definições Pydantic compartilhadas entre todos os módulos.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enums de Domínio
# =============================================================================

class Mercado(str, Enum):
    BR = "BR"        # B3
    US = "US"        # NYSE/NASDAQ
    FUTURES = "FUT"  # Futuros/Commodities
    CRYPTO = "CRYPTO"


class TipoSinal(str, Enum):
    COMPRA = "COMPRA"
    VENDA = "VENDA"
    HOLD = "HOLD"
    REBALANCEAMENTO = "REBALANCEAMENTO"


class StatusSinal(str, Enum):
    GERADO = "GERADO"
    VALIDADO = "VALIDADO"
    REJEITADO = "REJEITADO"
    EXECUTADO = "EXECUTADO"
    EXPIRADO = "EXPIRADO"


class FonteSinal(str, Enum):
    LSTM = "LSTM"
    GARCH = "GARCH"
    MONTECARLO = "MONTECARLO"
    MARKOWITZ = "MARKOWITZ"
    TRADINGVIEW = "TRADINGVIEW"
    MANUAL = "MANUAL"
    COMPOSTO = "COMPOSTO"


class NivelRisco(str, Enum):
    BAIXO = "BAIXO"
    MEDIO = "MEDIO"
    ALTO = "ALTO"
    CRITICO = "CRITICO"


class ClasseAtivo(str, Enum):
    """Classe do ativo para classificação de carteira."""
    ACAO = "ACAO"
    RENDA_FIXA = "RENDA_FIXA"
    FII = "FII"
    ETF = "ETF"
    FUTURO = "FUTURO"
    OPCAO = "OPCAO"
    CRYPTO = "CRYPTO"


class TipoRendaFixa(str, Enum):
    """Tipo de título de renda fixa."""
    PREFIXADO = "PREFIXADO"
    IPCA_MAIS = "IPCA+"
    SELIC = "SELIC"
    CDI = "CDI"
    CDI_MAIS = "CDI+"
    DEBENTURE = "DEBENTURE"


class RatingCredito(str, Enum):
    """Rating de crédito para títulos."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"


class RecomendacaoValuation(str, Enum):
    """Recomendação qualitativa de valuation."""
    SUBVALORIZADO = "SUBVALORIZADO"
    JUSTO = "JUSTO"
    SOBREVALORIZADO = "SOBREVALORIZADO"


# =============================================================================
# Modelos de Dados
# =============================================================================

class Ativo(BaseModel):
    """Representação de um ativo financeiro."""
    ticker: str
    nome: Optional[str] = None
    mercado: Mercado = Mercado.BR
    classe: ClasseAtivo = ClasseAtivo.ACAO
    setor: Optional[str] = None
    sufixo_yf: str = ""  # Ex: ".SA" para B3

    @property
    def ticker_yfinance(self) -> str:
        return f"{self.ticker}{self.sufixo_yf}"


class TituloRendaFixaModel(BaseModel):
    """Título de renda fixa (Pydantic model para API/serialização)."""
    nome: str
    tipo: TipoRendaFixa
    valor_nominal: float = 1000.0
    taxa_cupom: float = 0.0
    taxa_mercado: float = 0.10
    vencimento: datetime
    emissao: Optional[datetime] = None
    rating: Optional[RatingCredito] = None
    indexador: Optional[str] = None
    spread: float = 0.0
    quantidade: float = 1.0


class ResultadoMtMModel(BaseModel):
    """Resultado de marcação a mercado (para serialização)."""
    titulo: str
    pu_curva: float
    pu_mercado: float
    diferenca_absoluta: float
    diferenca_percentual: float
    duration_modificada: float
    dv01: float


class ResultadoValuationModel(BaseModel):
    """Resultado de valuation de ação (para serialização via API)."""
    ticker: str
    data_referencia: datetime
    preco_atual: float

    # Fair values por método
    dcf_fair_value: Optional[float] = None
    ddm_fair_value: Optional[float] = None
    graham_number: Optional[float] = None
    multiplos_fair_value: Optional[float] = None
    fair_value_composto: Optional[float] = None
    upside_downside: Optional[float] = None

    # Scores
    piotroski_f_score: Optional[int] = None
    altman_z_score: Optional[float] = None

    # Múltiplos
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    earnings_yield: Optional[float] = None
    fcf_yield: Optional[float] = None
    dividend_yield: Optional[float] = None

    # Recomendação
    recomendacao: RecomendacaoValuation = RecomendacaoValuation.JUSTO
    confianca: float = 0.0


class Sinal(BaseModel):
    """Sinal gerado pelo motor quantitativo."""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Identificação
    ativo: Ativo
    tipo: TipoSinal
    fonte: FonteSinal
    status: StatusSinal = StatusSinal.GERADO
    
    # Dados do sinal
    preco_referencia: float
    preco_alvo: Optional[float] = None
    stop_loss: Optional[float] = None
    peso_sugerido: Optional[float] = None  # % na carteira
    confianca: float = 0.5  # 0.0 a 1.0
    
    # Risco
    nivel_risco: NivelRisco = NivelRisco.MEDIO
    var_estimado: Optional[float] = None
    drawdown_maximo: Optional[float] = None
    
    # Evidência
    motivo: str = ""
    metricas: dict[str, Any] = Field(default_factory=dict)
    modelo_versao: Optional[str] = None
    
    # Governança
    aprovado_por: Optional[str] = None
    aprovado_em: Optional[datetime] = None
    motivo_rejeicao: Optional[str] = None


class ResultadoBacktest(BaseModel):
    """Resultado de um backtest."""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    estrategia: str
    periodo_inicio: datetime
    periodo_fim: datetime
    
    # Métricas
    retorno_total: float
    retorno_anualizado: float
    volatilidade_anualizada: float
    sharpe_ratio: float
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    calmar_ratio: Optional[float] = None
    
    # VaR
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    
    # Trade stats
    total_trades: int = 0
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Metadados
    parametros: dict[str, Any] = Field(default_factory=dict)
    notas: str = ""


class WebhookPayload(BaseModel):
    """Payload recebido do TradingView webhook."""
    id: UUID = Field(default_factory=uuid4)
    received_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Dados do alerta TradingView
    ticker: str
    action: str  # "buy", "sell", "alert"
    price: float
    interval: Optional[str] = None
    exchange: Optional[str] = None
    
    # Dados customizados do Pine Script
    strategy_name: Optional[str] = None
    indicator_value: Optional[float] = None
    message: Optional[str] = None
    
    # Metadados de segurança
    source_ip: Optional[str] = None
    webhook_token: Optional[str] = None
    idempotency_key: Optional[str] = None
    
    # Processamento
    is_valid: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    processed: bool = False
    sinal_gerado_id: Optional[UUID] = None


class MetricaPortfolio(BaseModel):
    """Métricas do portfolio em tempo real."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Composição
    ativos: dict[str, float]  # ticker → peso
    valor_total: float
    
    # Performance
    retorno_dia: float
    retorno_mtd: float
    retorno_ytd: float
    retorno_acumulado: float
    
    # Risco
    volatilidade_30d: float
    sharpe_30d: float
    var_95: float
    max_drawdown: float
    
    # Sinais pendentes
    sinais_pendentes: int = 0
    alertas_nao_validados: int = 0
