"""
QuantNucleo — Exemplo: Estratégia Momentum (MM Crossover)
==========================================================
Estratégia clássica de cruzamento de médias móveis.
Serve como template e referência para novas estratégias.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from libs.core.types import (
    Ativo,
    FonteSinal,
    Mercado,
    ResultadoBacktest,
    Sinal,
    TipoSinal,
)
from libs.risk.engine import RiskEngine
from strategies.base import StrategyBase, StrategyConfig


class MomentumCrossover(StrategyBase):
    """
    Estratégia: Cruzamento de Médias Móveis (Golden/Death Cross).
    
    - Golden Cross (MM curta cruza acima da longa) → COMPRA
    - Death Cross (MM curta cruza abaixo da longa) → VENDA
    """

    def __init__(
        self,
        tickers: list[str],
        mm_curta: int = 20,
        mm_longa: int = 50,
    ) -> None:
        config = StrategyConfig(
            nome="MomentumCrossover",
            versao="0.1.0",
            descricao=f"Cruzamento MM{mm_curta}/MM{mm_longa}",
            tickers=tickers,
            parametros={"mm_curta": mm_curta, "mm_longa": mm_longa},
        )
        super().__init__(config)
        self.mm_curta = mm_curta
        self.mm_longa = mm_longa
        self.risk_engine = RiskEngine()

    def generate_signals(
        self,
        precos: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Sinal]:
        """Gera sinais de cruzamento para cada ativo."""
        sinais: list[Sinal] = []

        for col in precos.columns:
            serie = precos[col].dropna()
            if len(serie) < self.mm_longa + 5:
                continue

            mm_c = serie.rolling(self.mm_curta).mean()
            mm_l = serie.rolling(self.mm_longa).mean()

            # Verificar cruzamento no último dia
            if mm_c.iloc[-1] > mm_l.iloc[-1] and mm_c.iloc[-2] <= mm_l.iloc[-2]:
                tipo = TipoSinal.COMPRA
                motivo = "Golden Cross"
            elif mm_c.iloc[-1] < mm_l.iloc[-1] and mm_c.iloc[-2] >= mm_l.iloc[-2]:
                tipo = TipoSinal.VENDA
                motivo = "Death Cross"
            else:
                continue

            sufixo = ".SA" if col.endswith(".SA") else ""
            ticker_limpo = col.replace(".SA", "")
            mercado = Mercado.BR if sufixo else Mercado.US

            ativo = Ativo(ticker=ticker_limpo, mercado=mercado, sufixo_yf=sufixo)
            retornos = serie.pct_change().dropna()

            sinais.append(Sinal(
                ativo=ativo,
                tipo=tipo,
                fonte=FonteSinal.COMPOSTO,
                preco_referencia=float(serie.iloc[-1]),
                confianca=abs(float(mm_c.iloc[-1] - mm_l.iloc[-1])) / float(serie.iloc[-1]),
                var_estimado=self.risk_engine.var_historico(retornos),
                drawdown_maximo=self.risk_engine.max_drawdown(retornos),
                motivo=f"{motivo} MM{self.mm_curta}/MM{self.mm_longa}",
            ))

        return sinais

    def backtest(
        self,
        precos: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        capital_inicial: float = 10_000_000.0,
    ) -> ResultadoBacktest:
        """Backtest simples de cruzamento de médias."""
        # Portfolio equi-ponderado
        retornos = precos.pct_change().dropna()
        n_ativos = len(precos.columns)
        pesos = np.ones(n_ativos) / n_ativos

        # Sinais: posição = 1 se mm_curta > mm_longa, 0 caso contrário
        posicoes = pd.DataFrame(0.0, index=precos.index, columns=precos.columns)
        for col in precos.columns:
            mm_c = precos[col].rolling(self.mm_curta).mean()
            mm_l = precos[col].rolling(self.mm_longa).mean()
            posicoes[col] = (mm_c > mm_l).astype(float)

        # Retorno da estratégia
        ret_estrategia = (posicoes.shift(1) * retornos).sum(axis=1) / n_ativos
        ret_estrategia = ret_estrategia.dropna()

        # Métricas
        retorno_total = float((1 + ret_estrategia).prod() - 1)
        n_anos = len(ret_estrategia) / 252
        retorno_anual = float((1 + retorno_total) ** (1 / max(n_anos, 0.01)) - 1)
        vol_anual = float(ret_estrategia.std() * np.sqrt(252))
        sharpe = retorno_anual / vol_anual if vol_anual > 0 else 0

        # Drawdown
        cumret = (1 + ret_estrategia).cumprod()
        max_dd = float((cumret / cumret.cummax() - 1).min())

        # Trades (mudanças de posição)
        trades = (posicoes.diff().abs() > 0).sum().sum()

        return ResultadoBacktest(
            estrategia=self.config.nome,
            periodo_inicio=precos.index[0].to_pydatetime(),
            periodo_fim=precos.index[-1].to_pydatetime(),
            retorno_total=retorno_total,
            retorno_anualizado=retorno_anual,
            volatilidade_anualizada=vol_anual,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=int(trades),
            parametros=self.config.parametros,
        )
