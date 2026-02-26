"""
QuantNucleo — Testes Unitários: Motor de Risco
================================================
"""

import unittest

import numpy as np
import pandas as pd

from libs.risk.engine import RiskEngine
from libs.core.types import Ativo, FonteSinal, Mercado, NivelRisco, Sinal, TipoSinal


class TestRiskEngine(unittest.TestCase):
    """Testes do motor de risco."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.engine = RiskEngine()
        # Gerar retornos simulados (252 dias, 5 ativos)
        self.retornos = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (252, 5)),
            columns=["ATIVO_A", "ATIVO_B", "ATIVO_C", "ATIVO_D", "ATIVO_E"],
        )
        self.pesos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def test_var_historico(self) -> None:
        """VaR histórico deve ser negativo para retornos normais."""
        var = self.engine.var_historico(self.retornos["ATIVO_A"])
        self.assertLess(var, 0)

    def test_var_parametrico(self) -> None:
        """VaR paramétrico deve ser negativo."""
        var = self.engine.var_parametrico(self.retornos["ATIVO_A"])
        self.assertLess(var, 0)

    def test_cvar(self) -> None:
        """CVaR deve ser menor (mais negativo) que VaR."""
        var = self.engine.var_historico(self.retornos["ATIVO_A"])
        cvar = self.engine.cvar(self.retornos["ATIVO_A"])
        self.assertLessEqual(cvar, var)

    def test_monte_carlo(self) -> None:
        """Monte Carlo deve retornar resultado estruturado."""
        resultado = self.engine.monte_carlo(
            self.retornos, self.pesos,
            simulacoes=100, horizonte=63,
        )
        self.assertIn("valor_final_medio", resultado)
        self.assertIn("var_95", resultado)
        self.assertIn("prob_perda", resultado)
        self.assertGreater(resultado["valor_final_medio"], 0)

    def test_max_drawdown(self) -> None:
        """Max drawdown deve ser negativo ou zero."""
        mdd = self.engine.max_drawdown(self.retornos["ATIVO_A"])
        self.assertLessEqual(mdd, 0)

    def test_sharpe_ratio(self) -> None:
        """Sharpe ratio deve ser finito."""
        sharpe = self.engine.sharpe_ratio(self.retornos["ATIVO_A"])
        self.assertTrue(np.isfinite(sharpe))

    def test_sortino_ratio(self) -> None:
        """Sortino ratio deve ser finito."""
        sortino = self.engine.sortino_ratio(self.retornos["ATIVO_A"])
        self.assertTrue(np.isfinite(sortino))

    def test_validar_sinal_aprovado(self) -> None:
        """Sinal com parâmetros normais deve ser aprovado."""
        sinal = Sinal(
            ativo=Ativo(ticker="PETR4", mercado=Mercado.BR, sufixo_yf=".SA"),
            tipo=TipoSinal.COMPRA,
            fonte=FonteSinal.COMPOSTO,
            preco_referencia=38.50,
            peso_sugerido=0.10,
            confianca=0.7,
            var_estimado=-0.03,
            drawdown_maximo=-0.08,
        )
        aprovado, rejeicoes = self.engine.validar_sinal(sinal)
        self.assertTrue(aprovado)
        self.assertEqual(len(rejeicoes), 0)

    def test_validar_sinal_rejeitado_peso(self) -> None:
        """Sinal com peso excessivo deve ser rejeitado."""
        sinal = Sinal(
            ativo=Ativo(ticker="PETR4", mercado=Mercado.BR, sufixo_yf=".SA"),
            tipo=TipoSinal.COMPRA,
            fonte=FonteSinal.COMPOSTO,
            preco_referencia=38.50,
            peso_sugerido=0.50,  # Excede limite de 30%
            confianca=0.7,
        )
        aprovado, rejeicoes = self.engine.validar_sinal(sinal)
        self.assertFalse(aprovado)
        self.assertGreater(len(rejeicoes), 0)

    def test_relatorio_risco(self) -> None:
        """Relatório de risco deve conter todas as métricas."""
        relatorio = self.engine.relatorio_risco(self.retornos, self.pesos)
        self.assertIn("sharpe", relatorio)
        self.assertIn("sortino", relatorio)
        self.assertIn("max_drawdown", relatorio)
        self.assertIn("var_95_historico", relatorio)
        self.assertIn("por_ativo", relatorio)


if __name__ == "__main__":
    unittest.main()
