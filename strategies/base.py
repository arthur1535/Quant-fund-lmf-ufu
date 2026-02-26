"""
QuantNucleo — Strategy Base Class
====================================
Classe abstrata para todas as estratégias de trading.

Toda estratégia DEVE:
 - Herdar de StrategyBase
 - Implementar generate_signals()
 - Implementar backtest()
 - Ter testes unitários e de backtest

Design:
 - research/ contém protótipos experimentais
 - strategies/ contém versões PROMOVIDAS para produção
 - Promoção requer: testes passando, backtest validado, code review
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from libs.core.types import ResultadoBacktest, Sinal


@dataclass
class StrategyConfig:
    """Configuração base de uma estratégia."""
    nome: str
    versao: str = "0.1.0"
    descricao: str = ""
    autor: str = ""
    tickers: list[str] = field(default_factory=list)
    parametros: dict[str, Any] = field(default_factory=dict)
    
    # Limites de risco
    max_posicao: float = 0.30
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.10
    max_drawdown: float = -0.15


class StrategyBase(ABC):
    """
    Classe abstrata base para estratégias.
    
    Todas as estratégias devem herdar dessa classe e implementar
    os métodos abstratos.
    
    Uso:
        class MinhaEstrategia(StrategyBase):
            def generate_signals(self, precos, features):
                ...
            def backtest(self, precos, features):
                ...
        
        est = MinhaEstrategia(config)
        sinais = est.generate_signals(precos, features)
        resultado = est.backtest(precos, features)
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self._sinais: list[Sinal] = []

    @abstractmethod
    def generate_signals(
        self,
        precos: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Sinal]:
        """
        Gera sinais a partir de dados de mercado.
        
        Args:
            precos: DataFrame de preços (Close)
            features: DataFrame de features computadas
            
        Returns:
            Lista de sinais gerados
        """
        ...

    @abstractmethod
    def backtest(
        self,
        precos: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        capital_inicial: float = 10_000_000.0,
    ) -> ResultadoBacktest:
        """
        Executa backtest da estratégia.
        
        Args:
            precos: DataFrame de preços históricos
            features: DataFrame de features
            capital_inicial: Capital inicial em R$
            
        Returns:
            Resultado do backtest
        """
        ...

    def validate(self, resultado: ResultadoBacktest) -> tuple[bool, list[str]]:
        """
        Validação estatística mínima do backtest.
        Critérios:
         - Sharpe > 0
         - Max drawdown não viola limite
         - Número mínimo de trades
        """
        erros: list[str] = []

        if resultado.sharpe_ratio < 0:
            erros.append(f"Sharpe negativo: {resultado.sharpe_ratio:.2f}")

        if resultado.max_drawdown < self.config.max_drawdown:
            erros.append(
                f"Max drawdown ({resultado.max_drawdown:.1%}) viola limite "
                f"({self.config.max_drawdown:.1%})"
            )

        if resultado.total_trades < 10:
            erros.append(f"Poucos trades: {resultado.total_trades} (mín: 10)")

        return len(erros) == 0, erros

    @property
    def info(self) -> dict[str, Any]:
        """Informações da estratégia."""
        return {
            "nome": self.config.nome,
            "versao": self.config.versao,
            "descricao": self.config.descricao,
            "autor": self.config.autor,
            "tickers": self.config.tickers,
            "parametros": self.config.parametros,
        }
