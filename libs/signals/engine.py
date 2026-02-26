"""
QuantNucleo — Motor de Sinais
===============================
Gera sinais quantitativos a partir de múltiplos modelos/fontes.

Pipeline:
  1. Recebe dados (preços, features, alertas TV)
  2. Aplica modelos (LSTM, GARCH, Momentum, etc.)
  3. Combina sinais (ensemble / votação ponderada)
  4. Valida contra motor de risco
  5. Registra no MongoDB (auditoria)
  6. Emite notificação / ação

A fonte de verdade é SEMPRE o núcleo quant, não o TradingView.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from libs.core.logging import get_logger
from libs.core.types import (
    Ativo,
    FonteSinal,
    NivelRisco,
    Sinal,
    StatusSinal,
    TipoSinal,
)
from libs.risk.engine import RiskEngine

logger = get_logger("signals.engine")


class SignalEngine:
    """
    Motor de sinais do núcleo quant.
    Centraliza geração, validação e registro de sinais.
    """

    def __init__(self, risk_engine: Optional[RiskEngine] = None) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        self._sinais_pendentes: list[Sinal] = []

    # =========================================================================
    # Geração de Sinais
    # =========================================================================

    def gerar_sinal_momentum(
        self,
        ativo: Ativo,
        precos: pd.Series,
        janela_curta: int = 20,
        janela_longa: int = 50,
    ) -> Optional[Sinal]:
        """
        Sinal baseado em cruzamento de médias móveis.
        
        Golden Cross → COMPRA
        Death Cross  → VENDA
        """
        if len(precos) < janela_longa + 5:
            return None

        mm_curta = precos.rolling(janela_curta).mean()
        mm_longa = precos.rolling(janela_longa).mean()

        # Detectar cruzamento recente (últimos 2 dias)
        sinal_hoje = mm_curta.iloc[-1] > mm_longa.iloc[-1]
        sinal_ontem = mm_curta.iloc[-2] > mm_longa.iloc[-2]

        if sinal_hoje == sinal_ontem:
            return None  # Sem cruzamento

        tipo = TipoSinal.COMPRA if sinal_hoje else TipoSinal.VENDA
        preco_atual = float(precos.iloc[-1])
        confianca = abs(float(mm_curta.iloc[-1] - mm_longa.iloc[-1])) / preco_atual

        retornos = precos.pct_change().dropna()
        var_95 = self.risk_engine.var_historico(retornos)

        sinal = Sinal(
            ativo=ativo,
            tipo=tipo,
            fonte=FonteSinal.COMPOSTO,
            preco_referencia=preco_atual,
            confianca=min(confianca * 10, 1.0),  # normalizar
            var_estimado=var_95,
            drawdown_maximo=self.risk_engine.max_drawdown(retornos),
            motivo=f"Cruzamento MM{janela_curta}/MM{janela_longa} — {'Golden Cross' if tipo == TipoSinal.COMPRA else 'Death Cross'}",
            metricas={
                "mm_curta": float(mm_curta.iloc[-1]),
                "mm_longa": float(mm_longa.iloc[-1]),
                "spread_pct": float((mm_curta.iloc[-1] - mm_longa.iloc[-1]) / mm_longa.iloc[-1]),
            },
        )

        logger.info(
            "sinal_momentum_gerado",
            ativo=ativo.ticker,
            tipo=tipo.value,
            confianca=sinal.confianca,
        )
        return sinal

    def gerar_sinal_rsi(
        self,
        ativo: Ativo,
        precos: pd.Series,
        janela: int = 14,
        sobrecompra: float = 70,
        sobrevenda: float = 30,
    ) -> Optional[Sinal]:
        """Sinal baseado em RSI extremo."""
        if len(precos) < janela + 5:
            return None

        delta = precos.diff()
        ganhos = delta.where(delta > 0, 0.0).rolling(janela).mean()
        perdas = (-delta.where(delta < 0, 0.0)).rolling(janela).mean()
        rs = ganhos / perdas.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_atual = float(rsi.iloc[-1])

        if rsi_atual > sobrecompra:
            tipo = TipoSinal.VENDA
        elif rsi_atual < sobrevenda:
            tipo = TipoSinal.COMPRA
        else:
            return None

        preco_atual = float(precos.iloc[-1])
        retornos = precos.pct_change().dropna()

        return Sinal(
            ativo=ativo,
            tipo=tipo,
            fonte=FonteSinal.COMPOSTO,
            preco_referencia=preco_atual,
            confianca=abs(rsi_atual - 50) / 50,
            var_estimado=self.risk_engine.var_historico(retornos),
            motivo=f"RSI({janela}) = {rsi_atual:.1f} — {'Sobrecompra' if tipo == TipoSinal.VENDA else 'Sobrevenda'}",
            metricas={"rsi": rsi_atual},
        )

    # =========================================================================
    # Processamento de Webhook TradingView
    # =========================================================================

    def processar_alerta_tradingview(
        self,
        ticker: str,
        action: str,
        price: float,
        message: Optional[str] = None,
        indicator_value: Optional[float] = None,
    ) -> Sinal:
        """
        Processa alerta do TradingView e cria sinal.
        
        IMPORTANTE: TradingView é canal de alerta, NÃO fonte de verdade.
        O sinal criado é marcado como GERADO e precisa validação.
        """
        ativo = Ativo(ticker=ticker.replace(".SA", ""), sufixo_yf=".SA" if ".SA" in ticker else "")

        tipo_map = {
            "buy": TipoSinal.COMPRA,
            "sell": TipoSinal.VENDA,
            "hold": TipoSinal.HOLD,
        }
        tipo = tipo_map.get(action.lower(), TipoSinal.HOLD)

        sinal = Sinal(
            ativo=ativo,
            tipo=tipo,
            fonte=FonteSinal.TRADINGVIEW,
            status=StatusSinal.GERADO,
            preco_referencia=price,
            confianca=0.5,  # Confiança base — será recalculada
            motivo=f"Alerta TradingView: {action} @ {price} — {message or 'sem mensagem'}",
            metricas={
                "tv_action": action,
                "tv_price": price,
                "tv_indicator": indicator_value,
            },
        )

        logger.info(
            "alerta_tv_processado",
            ticker=ticker,
            tipo=tipo.value,
            preco=price,
        )
        return sinal

    # =========================================================================
    # Validação e Governança
    # =========================================================================

    def validar_e_registrar(self, sinal: Sinal) -> Sinal:
        """
        Pipeline de validação:
        1. Passa pelo motor de risco
        2. Atualiza status
        3. Registra (pronto para auditoria MongoDB)
        """
        aprovado, rejeicoes = self.risk_engine.validar_sinal(sinal)

        if aprovado:
            sinal.status = StatusSinal.VALIDADO
        else:
            sinal.status = StatusSinal.REJEITADO
            sinal.motivo_rejeicao = "; ".join(rejeicoes)

        self._sinais_pendentes.append(sinal)

        logger.info(
            "sinal_validado",
            sinal_id=str(sinal.id),
            status=sinal.status.value,
            rejeicoes=rejeicoes,
        )
        return sinal

    @property
    def sinais_pendentes(self) -> list[Sinal]:
        """Sinais validados aguardando aprovação humana."""
        return [
            s for s in self._sinais_pendentes
            if s.status == StatusSinal.VALIDADO
        ]
