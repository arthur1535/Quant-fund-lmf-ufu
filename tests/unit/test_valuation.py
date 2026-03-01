"""
QuantNucleo — Testes Unitários: Motor de Valuation
====================================================
Testa DCF, DDM, Graham, Múltiplos, Piotroski, Altman.
"""

import unittest
from datetime import date

import numpy as np
import pandas as pd

from libs.valuation.engine import (
    DadosFundamentalistas,
    ResultadoValuation,
    ValuationEngine,
)


def _criar_dados_exemplo(
    ticker: str = "TEST4.SA",
    preco: float = 30.0,
) -> DadosFundamentalistas:
    """Cria dados fundamentalistas sintéticos para testes."""
    shares = 1_000_000_000
    return DadosFundamentalistas(
        ticker=ticker,
        data_referencia=date.today(),
        preco_atual=preco,
        market_cap=preco * shares,
        shares_outstanding=shares,
        enterprise_value=preco * shares + 5e9,
        receita_liquida=50e9,
        ebitda=15e9,
        ebit=12e9,
        lucro_liquido=8e9,
        lpa=8.0,
        ativo_total=100e9,
        ativo_circulante=25e9,
        passivo_total=60e9,
        passivo_circulante=15e9,
        patrimonio_liquido=40e9,
        divida_bruta=20e9,
        divida_liquida=10e9,
        caixa=10e9,
        capital_giro=10e9,
        fluxo_caixa_operacional=14e9,
        capex=4e9,
        fcf=10e9,
        dividendos_pagos=3e9,
        margem_bruta=0.45,
        margem_ebitda=0.30,
        margem_liquida=0.16,
        margem_fcf=0.20,
        roe=0.20,
        roa=0.08,
        roic=0.12,
        dividendo_por_acao=3.0,
        dividend_yield=0.10,
        payout_ratio=0.375,
        setor="Energy",
        moeda="BRL",
    )


class TestValuationEngine(unittest.TestCase):
    """Testes do motor de valuation."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.engine = ValuationEngine()
        self.dados = _criar_dados_exemplo()

    # =========================================================================
    # Testes de DCF
    # =========================================================================

    def test_dcf_fair_value_positivo(self) -> None:
        """DCF deve retornar fair value positivo."""
        fv = self.engine.dcf_fcff(self.dados)
        self.assertGreater(fv, 0)

    def test_dcf_fcf_negativo_retorna_zero(self) -> None:
        """DCF com FCF negativo deve retornar 0."""
        dados = _criar_dados_exemplo()
        dados.fcf = -5e9
        fv = self.engine.dcf_fcff(dados)
        self.assertEqual(fv, 0.0)

    def test_dcf_crescimento_maior_aumenta_fv(self) -> None:
        """Fair value deve aumentar com maior taxa de crescimento."""
        fv_baixo = self.engine.dcf_fcff(self.dados, taxa_crescimento_5y=0.02)
        fv_alto = self.engine.dcf_fcff(self.dados, taxa_crescimento_5y=0.10)
        self.assertGreater(fv_alto, fv_baixo)

    def test_dcf_wacc_maior_diminui_fv(self) -> None:
        """Fair value deve diminuir com WACC maior (mais desconto)."""
        fv_baixo_wacc = self.engine.dcf_fcff(self.dados, wacc=0.10)
        fv_alto_wacc = self.engine.dcf_fcff(self.dados, wacc=0.18)
        self.assertGreater(fv_baixo_wacc, fv_alto_wacc)

    def test_dcf_resultado_razoavel(self) -> None:
        """Fair value não deve ser absurdamente diferente do preço."""
        fv = self.engine.dcf_fcff(self.dados)
        # Deve estar entre 0.1x e 100x do preço (sanidade)
        ratio = fv / self.dados.preco_atual
        self.assertGreater(ratio, 0.1)
        self.assertLess(ratio, 100)

    # =========================================================================
    # Testes de DDM (Gordon Growth)
    # =========================================================================

    def test_ddm_fair_value_positivo(self) -> None:
        """DDM deve retornar fair value positivo para pagadoras de dividendos."""
        fv = self.engine.ddm_gordon(self.dados)
        self.assertGreater(fv, 0)

    def test_ddm_sem_dividendos_retorna_zero(self) -> None:
        """DDM sem dividendos deve retornar 0."""
        dados = _criar_dados_exemplo()
        dados.dividendo_por_acao = 0.0
        fv = self.engine.ddm_gordon(dados)
        self.assertEqual(fv, 0.0)

    def test_ddm_ke_menor_g_retorna_zero(self) -> None:
        """DDM com ke ≤ g deve retornar 0 (modelo não converge)."""
        fv = self.engine.ddm_gordon(
            self.dados,
            taxa_crescimento_dividendo=0.20,  # g > ke
            custo_capital_proprio=0.15,
        )
        self.assertEqual(fv, 0.0)

    def test_ddm_dois_estagios_positivo(self) -> None:
        """DDM dois estágios deve retornar valor positivo."""
        fv = self.engine.ddm_dois_estagios(self.dados, g_alto=0.08, g_perpetuo=0.04)
        self.assertGreater(fv, 0)

    # =========================================================================
    # Testes de Graham Number
    # =========================================================================

    def test_graham_positivo(self) -> None:
        """Graham Number deve ser positivo para EPS e BVPS positivos."""
        gn = self.engine.graham_number(self.dados)
        self.assertGreater(gn, 0)

    def test_graham_formula_correta(self) -> None:
        """Graham = sqrt(22.5 * EPS * BVPS)."""
        gn = self.engine.graham_number(self.dados)
        bvps = self.dados.valor_patrimonial_por_acao
        esperado = np.sqrt(22.5 * self.dados.lpa * bvps)
        self.assertAlmostEqual(gn, esperado, places=2)

    def test_graham_eps_negativo(self) -> None:
        """Graham deve ser 0 para EPS negativo."""
        dados = _criar_dados_exemplo()
        dados.lpa = -2.0
        gn = self.engine.graham_number(dados)
        self.assertEqual(gn, 0.0)

    # =========================================================================
    # Testes de Múltiplos
    # =========================================================================

    def test_multiplos_completos(self) -> None:
        """Deve calcular todos os múltiplos esperados."""
        mult = self.engine.calcular_multiplos(self.dados)
        self.assertIn("pe_ratio", mult)
        self.assertIn("pb_ratio", mult)
        self.assertIn("ev_ebitda", mult)
        self.assertIn("earnings_yield", mult)
        self.assertIn("fcf_yield", mult)
        self.assertIn("dividend_yield", mult)

    def test_pe_ratio_correto(self) -> None:
        """P/E = preço / LPA."""
        mult = self.engine.calcular_multiplos(self.dados)
        esperado = self.dados.preco_atual / self.dados.lpa
        self.assertAlmostEqual(mult["pe_ratio"], round(esperado, 2), places=2)

    def test_earnings_yield_inverso_pe(self) -> None:
        """Earnings Yield ≈ 1/PE."""
        mult = self.engine.calcular_multiplos(self.dados)
        pe = mult["pe_ratio"]
        ey = mult["earnings_yield"]
        if pe and ey:
            self.assertAlmostEqual(ey, round(1 / pe, 4), places=3)

    def test_multiplos_lpa_negativo(self) -> None:
        """PE deve ser None para LPA negativo (empresa prejuízo)."""
        dados = _criar_dados_exemplo()
        dados.lpa = -1.0
        mult = self.engine.calcular_multiplos(dados)
        self.assertIsNone(mult["pe_ratio"])

    def test_fair_value_multiplos(self) -> None:
        """Fair value por múltiplos deve ser positivo."""
        fv = self.engine.fair_value_por_multiplos(self.dados, pe_justo=12, ev_ebitda_justo=8)
        self.assertGreater(fv, 0)

    # =========================================================================
    # Testes de Piotroski F-Score
    # =========================================================================

    def test_piotroski_range_0_9(self) -> None:
        """Piotroski deve estar entre 0 e 9."""
        score = self.engine.piotroski_f_score(self.dados)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 9)

    def test_piotroski_empresa_forte(self) -> None:
        """Empresa com bons fundamentos deve ter score alto (≥ 6)."""
        score = self.engine.piotroski_f_score(self.dados)
        self.assertGreaterEqual(score, 6, "Empresa saudável deveria ter Piotroski ≥ 6")

    def test_piotroski_empresa_fraca(self) -> None:
        """Empresa com fundamentos ruins deve ter score baixo."""
        dados_ruins = DadosFundamentalistas(
            ticker="RUIM3.SA",
            data_referencia=date.today(),
            preco_atual=5.0,
            roa=-0.05,
            fluxo_caixa_operacional=-1e9,
            lucro_liquido=-2e9,
            divida_bruta=80e9,
            ativo_total=100e9,
            ativo_circulante=10e9,
            passivo_circulante=20e9,
            shares_outstanding=1e9,
            margem_bruta=0.05,
            receita_liquida=20e9,
        )
        score = self.engine.piotroski_f_score(dados_ruins)
        self.assertLessEqual(score, 3)

    def test_piotroski_com_dados_anteriores(self) -> None:
        """Piotroski com dados do período anterior funciona."""
        anterior = _criar_dados_exemplo()
        anterior.roa = 0.15  # Menor que atual (0.20)
        score = self.engine.piotroski_f_score(self.dados, anterior)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 9)

    # =========================================================================
    # Testes de Altman Z-Score
    # =========================================================================

    def test_altman_finito(self) -> None:
        """Z-score deve ser finito."""
        z = self.engine.altman_z_score(self.dados)
        self.assertTrue(np.isfinite(z))

    def test_altman_empresa_saudavel(self) -> None:
        """Empresa saudável deve ter Z > 1.81 (fora da zona de perigo)."""
        z = self.engine.altman_z_score(self.dados)
        self.assertGreater(z, 1.81)

    def test_altman_ativo_zero(self) -> None:
        """Deve retornar 0 se ativo total é zero."""
        dados = _criar_dados_exemplo()
        dados.ativo_total = 0
        z = self.engine.altman_z_score(dados)
        self.assertEqual(z, 0.0)

    # =========================================================================
    # Testes de WACC
    # =========================================================================

    def test_wacc_positivo(self) -> None:
        """WACC deve ser positivo."""
        wacc = self.engine.calcular_wacc(self.dados)
        self.assertGreater(wacc, 0)

    def test_wacc_entre_ke_e_kd(self) -> None:
        """WACC deve estar entre custo do equity e custo da dívida."""
        rf, erp, kd = 0.1175, 0.06, 0.12
        ke = rf + 1.0 * erp  # CAPM com beta=1
        wacc = self.engine.calcular_wacc(
            self.dados, rf=rf, erp=erp, custo_divida=kd
        )
        kd_pos_ir = kd * (1 - 0.34)
        self.assertGreaterEqual(wacc, min(ke, kd_pos_ir) - 0.01)
        self.assertLessEqual(wacc, max(ke, kd_pos_ir) + 0.01)

    def test_custo_capital_capm(self) -> None:
        """CAPM: ke = rf + β × ERP."""
        ke = self.engine.custo_capital_capm(self.dados, rf=0.10, erp=0.06, beta=1.2)
        esperado = 0.10 + 1.2 * 0.06
        self.assertAlmostEqual(ke, esperado, places=6)

    # =========================================================================
    # Testes de Valuation Completo
    # =========================================================================

    def test_valuation_completo_retorna_resultado(self) -> None:
        """Valuation completo deve retornar ResultadoValuation."""
        resultado = self.engine.valuation_completo(self.dados)
        self.assertIsInstance(resultado, ResultadoValuation)
        self.assertEqual(resultado.ticker, self.dados.ticker)

    def test_valuation_completo_fair_value(self) -> None:
        """Fair value composto deve ser calculado."""
        resultado = self.engine.valuation_completo(self.dados)
        self.assertIsNotNone(resultado.fair_value_composto)
        self.assertGreater(resultado.fair_value_composto, 0)

    def test_valuation_completo_recomendacao(self) -> None:
        """Recomendação deve ser uma das três válidas."""
        resultado = self.engine.valuation_completo(self.dados)
        self.assertIn(
            resultado.recomendacao,
            ["SUBVALORIZADO", "JUSTO", "SOBREVALORIZADO"],
        )

    def test_valuation_completo_piotroski_presente(self) -> None:
        """Piotroski F-Score deve estar presente no resultado."""
        resultado = self.engine.valuation_completo(self.dados)
        self.assertIsNotNone(resultado.piotroski_f_score)

    def test_valuation_completo_altman_presente(self) -> None:
        """Altman Z-Score deve estar presente no resultado."""
        resultado = self.engine.valuation_completo(self.dados)
        self.assertIsNotNone(resultado.altman_z_score)

    def test_valuation_completo_upside(self) -> None:
        """Upside/downside deve ser calculado."""
        resultado = self.engine.valuation_completo(self.dados)
        self.assertIsNotNone(resultado.upside_downside)

    # =========================================================================
    # Testes de Ranking Relativo
    # =========================================================================

    def test_ranking_relativo(self) -> None:
        """Ranking relativo deve retornar DataFrame ordenado."""
        dados_list = [
            _criar_dados_exemplo("PETR4.SA", preco=35.0),
            _criar_dados_exemplo("VALE3.SA", preco=60.0),
            _criar_dados_exemplo("BBAS3.SA", preco=28.0),
        ]
        df = self.engine.ranking_relativo(dados_list)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("ticker", df.columns)
        self.assertIn("score_composto", df.columns)

    def test_ranking_lista_vazia(self) -> None:
        """Ranking com lista vazia deve retornar DataFrame vazio."""
        df = self.engine.ranking_relativo([])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)


class TestDadosFundamentalistas(unittest.TestCase):
    """Testes do dataclass DadosFundamentalistas."""

    def test_fcf_por_acao(self) -> None:
        """FCF por ação = FCF / shares."""
        dados = _criar_dados_exemplo()
        esperado = dados.fcf / dados.shares_outstanding
        self.assertAlmostEqual(dados.fcf_por_acao, esperado, places=4)

    def test_valor_patrimonial_por_acao(self) -> None:
        """VPA = PL / shares."""
        dados = _criar_dados_exemplo()
        esperado = dados.patrimonio_liquido / dados.shares_outstanding
        self.assertAlmostEqual(dados.valor_patrimonial_por_acao, esperado, places=4)

    def test_fcf_por_acao_zero_shares(self) -> None:
        """Deve retornar 0 se shares = 0."""
        dados = _criar_dados_exemplo()
        dados.shares_outstanding = 0
        self.assertEqual(dados.fcf_por_acao, 0.0)


if __name__ == "__main__":
    unittest.main()
