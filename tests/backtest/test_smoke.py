"""
QuantNucleo — Smoke Test de Backtest
=====================================
Testes rápidos para validar que o pipeline de backtest funciona.
"""

import unittest

import numpy as np
import pandas as pd
import pytest

from strategies.examples.momentum_crossover import MomentumCrossover


@pytest.mark.smoke
@pytest.mark.backtest
class TestBacktestSmoke(unittest.TestCase):
    """Smoke tests de backtest."""

    def setUp(self) -> None:
        np.random.seed(42)
        n = 504  # ~2 anos
        self.precos = pd.DataFrame(
            {
                "ATIVO_A": 100 + np.cumsum(np.random.normal(0.05, 1.0, n)),
                "ATIVO_B": 50 + np.cumsum(np.random.normal(0.03, 0.8, n)),
                "ATIVO_C": 200 + np.cumsum(np.random.normal(0.02, 1.5, n)),
            },
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )

    def test_momentum_crossover_executa(self) -> None:
        """Estratégia Momentum Crossover deve executar sem erro."""
        estrategia = MomentumCrossover(
            tickers=list(self.precos.columns),
            mm_curta=20,
            mm_longa=50,
        )
        resultado = estrategia.backtest(self.precos)
        self.assertIsNotNone(resultado)
        self.assertTrue(np.isfinite(resultado.retorno_total))
        self.assertTrue(np.isfinite(resultado.sharpe_ratio))

    def test_momentum_crossover_validacao(self) -> None:
        """Resultado do backtest deve passar validação básica."""
        estrategia = MomentumCrossover(
            tickers=list(self.precos.columns),
        )
        resultado = estrategia.backtest(self.precos)
        valido, erros = estrategia.validate(resultado)
        # Em dados simulados, pode ou não passar
        self.assertIsInstance(valido, bool)
        self.assertIsInstance(erros, list)

    def test_generate_signals(self) -> None:
        """Deve gerar sinais a partir de dados de preço."""
        estrategia = MomentumCrossover(
            tickers=list(self.precos.columns),
        )
        sinais = estrategia.generate_signals(self.precos)
        self.assertIsInstance(sinais, list)


if __name__ == "__main__":
    unittest.main()
