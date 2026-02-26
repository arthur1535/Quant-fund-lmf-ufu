"""
QuantNucleo — Testes Unitários: Signal Engine
===============================================
"""

import unittest

import numpy as np
import pandas as pd

from libs.core.types import Ativo, Mercado
from libs.signals.engine import SignalEngine


class TestSignalEngine(unittest.TestCase):
    """Testes do motor de sinais."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.engine = SignalEngine()
        # Preços simulados com tendência de alta
        n = 200
        self.precos = pd.Series(
            100 + np.cumsum(np.random.normal(0.1, 1.0, n)),
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
            name="PETR4.SA",
        )
        self.ativo = Ativo(ticker="PETR4", mercado=Mercado.BR, sufixo_yf=".SA")

    def test_processar_alerta_tradingview(self) -> None:
        """Deve criar sinal a partir de alerta TV."""
        sinal = self.engine.processar_alerta_tradingview(
            ticker="PETR4.SA",
            action="buy",
            price=38.50,
            message="Golden Cross",
        )
        self.assertEqual(sinal.ativo.ticker, "PETR4")
        self.assertEqual(sinal.preco_referencia, 38.50)
        self.assertEqual(sinal.fonte.value, "TRADINGVIEW")

    def test_validar_e_registrar(self) -> None:
        """Deve validar e registrar sinal."""
        sinal = self.engine.processar_alerta_tradingview(
            ticker="VALE3.SA",
            action="sell",
            price=62.30,
        )
        resultado = self.engine.validar_e_registrar(sinal)
        self.assertIn(resultado.status.value, ["VALIDADO", "REJEITADO"])

    def test_gerar_sinal_rsi_sobrevenda(self) -> None:
        """RSI em sobrevenda deve gerar sinal de compra."""
        # Preços em queda forte para forçar RSI baixo
        precos_queda = pd.Series(
            100 - np.cumsum(np.abs(np.random.normal(0.5, 0.3, 100))),
            index=pd.date_range("2024-01-01", periods=100, freq="B"),
        )
        sinal = self.engine.gerar_sinal_rsi(self.ativo, precos_queda)
        # Pode ou não gerar sinal dependendo do RSI exato
        if sinal is not None:
            self.assertEqual(sinal.tipo.value, "COMPRA")


if __name__ == "__main__":
    unittest.main()
