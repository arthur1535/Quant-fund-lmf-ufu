"""
QuantNucleo — Testes Unitários: Motor de Renda Fixa
=====================================================
Testa cálculos de PU, Duration, Convexidade, MtM e Z-spread.
"""

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from libs.fixed_income.titulos import (
    FluxoCaixa,
    IndexadorRF,
    RatingCredito,
    RendaFixaEngine,
    ResultadoMtM,
    TipoTitulo,
    TituloRendaFixa,
)


class TestRendaFixaEngine(unittest.TestCase):
    """Testes do motor de renda fixa."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.engine = RendaFixaEngine()
        self.hoje = date.today()

        # Título prefixado zero-coupon (LTN)
        self.ltn = TituloRendaFixa(
            nome="LTN 2026",
            tipo=TipoTitulo.PREFIXADO,
            valor_nominal=1000.0,
            taxa_cupom=0.0,
            taxa_mercado=0.12,
            vencimento=self.hoje + timedelta(days=504),  # ~2 anos úteis
            emissao=self.hoje - timedelta(days=252),
        )

        # Título IPCA+ com cupom
        self.ntnb = TituloRendaFixa(
            nome="NTN-B 2030",
            tipo=TipoTitulo.IPCA_MAIS,
            valor_nominal=1000.0,
            taxa_cupom=0.06,  # 6% a.a. cupom
            taxa_mercado=0.055,
            vencimento=self.hoje + timedelta(days=1260),  # ~5 anos úteis
            emissao=self.hoje - timedelta(days=252),
            indexador=IndexadorRF.IPCA,
            spread=0.055,
        )

        # Debênture CDI+
        self.debenture = TituloRendaFixa(
            nome="DEB XYZ CDI+2%",
            tipo=TipoTitulo.CDI_MAIS,
            valor_nominal=1000.0,
            taxa_cupom=0.0,
            taxa_mercado=0.13,
            vencimento=self.hoje + timedelta(days=756),  # ~3 anos úteis
            emissao=self.hoje - timedelta(days=126),
            indexador=IndexadorRF.CDI,
            spread=0.02,
            rating=RatingCredito.AA,
        )

    # =========================================================================
    # Testes de geração de fluxos
    # =========================================================================

    def test_gerar_fluxos_ltn_unico(self) -> None:
        """LTN (zero-coupon) deve gerar exatamente 1 fluxo."""
        fluxos = self.engine.gerar_fluxos(self.ltn)
        self.assertEqual(len(fluxos), 1)
        self.assertAlmostEqual(fluxos[0].valor, 1000.0, places=2)

    def test_gerar_fluxos_ntnb_multiplos(self) -> None:
        """NTN-B com cupom deve gerar múltiplos fluxos semestrais."""
        fluxos = self.engine.gerar_fluxos(self.ntnb)
        self.assertGreater(len(fluxos), 1)
        # Último fluxo inclui principal + cupom
        self.assertGreater(fluxos[-1].valor, 1000.0)

    def test_fluxo_datas_crescentes(self) -> None:
        """Datas dos fluxos devem ser estritamente crescentes."""
        fluxos = self.engine.gerar_fluxos(self.ntnb)
        for i in range(1, len(fluxos)):
            self.assertGreater(fluxos[i].data, fluxos[i - 1].data)

    # =========================================================================
    # Testes de PU (Preço Unitário)
    # =========================================================================

    def test_calcular_pu_positivo(self) -> None:
        """PU deve ser sempre positivo para taxas razoáveis."""
        pu = self.engine.calcular_pu(self.ltn)
        self.assertGreater(pu, 0)

    def test_pu_menor_que_nominal(self) -> None:
        """PU de título prefixado descontado deve ser menor que o nominal."""
        pu = self.engine.calcular_pu(self.ltn)
        self.assertLess(pu, self.ltn.valor_nominal)

    def test_pu_inversamente_proporcional_taxa(self) -> None:
        """PU deve diminuir quando taxa de desconto aumenta."""
        pu_baixa = self.engine.calcular_pu(self.ltn, taxa_override=0.10)
        pu_alta = self.engine.calcular_pu(self.ltn, taxa_override=0.15)
        self.assertGreater(pu_baixa, pu_alta)

    def test_pu_no_vencimento(self) -> None:
        """PU na data de vencimento deve ser próximo ao valor nominal."""
        titulo_venc = TituloRendaFixa(
            nome="LTN Vencendo",
            tipo=TipoTitulo.PREFIXADO,
            valor_nominal=1000.0,
            taxa_cupom=0.0,
            taxa_mercado=0.12,
            vencimento=self.hoje + timedelta(days=1),
        )
        pu = self.engine.calcular_pu(titulo_venc)
        self.assertAlmostEqual(pu, 1000.0, delta=5.0)

    # =========================================================================
    # Testes de YTM
    # =========================================================================

    def test_ytm_consistencia(self) -> None:
        """YTM calculado a partir do PU deve ser próximo da taxa original."""
        pu = self.engine.calcular_pu(self.ltn)
        ytm = self.engine.calcular_ytm(self.ltn, pu)
        self.assertAlmostEqual(ytm, self.ltn.taxa_mercado, places=3)

    def test_ytm_positivo(self) -> None:
        """YTM deve ser positivo para PU < VN (título com desconto)."""
        pu = self.engine.calcular_pu(self.ltn)
        ytm = self.engine.calcular_ytm(self.ltn, pu)
        self.assertGreater(ytm, 0)

    # =========================================================================
    # Testes de Duration
    # =========================================================================

    def test_duration_positiva(self) -> None:
        """Duration deve ser positiva."""
        dur = self.engine.duration_macaulay(self.ltn)
        self.assertGreater(dur, 0)

    def test_duration_zero_coupon_igual_prazo(self) -> None:
        """Duration de zero-coupon deve ser igual ao prazo em anos (DU/252)."""
        dur = self.engine.duration_macaulay(self.ltn)
        prazo_anos = 504 / 252  # ~2 anos
        self.assertAlmostEqual(dur, prazo_anos, delta=0.1)

    def test_duration_modificada_menor(self) -> None:
        """Duration modificada deve ser menor ou igual à Macaulay."""
        dur_mac = self.engine.duration_macaulay(self.ltn)
        dur_mod = self.engine.duration_modificada(self.ltn)
        self.assertLessEqual(dur_mod, dur_mac)

    def test_duration_coupon_menor_zero_coupon(self) -> None:
        """Duration de título com cupom deve ser menor que zero-coupon de mesmo prazo."""
        dur_ltn = self.engine.duration_macaulay(self.ltn)
        # Criar título com cupom e mesmo prazo
        titulo_cupom = TituloRendaFixa(
            nome="PREF COM CUPOM",
            tipo=TipoTitulo.PREFIXADO,
            valor_nominal=1000.0,
            taxa_cupom=0.10,
            taxa_mercado=0.12,
            vencimento=self.ltn.vencimento,
        )
        dur_cupom = self.engine.duration_macaulay(titulo_cupom)
        self.assertLess(dur_cupom, dur_ltn)

    # =========================================================================
    # Testes de Convexidade
    # =========================================================================

    def test_convexidade_positiva(self) -> None:
        """Convexidade deve ser positiva para bonds normais."""
        conv = self.engine.convexidade(self.ltn)
        self.assertGreater(conv, 0)

    # =========================================================================
    # Testes de DV01
    # =========================================================================

    def test_dv01_positivo(self) -> None:
        """DV01 (dollar value of 01bp) deve ser positivo."""
        dv01 = self.engine.dv01(self.ltn)
        self.assertGreater(dv01, 0)

    def test_dv01_proporcional_duration(self) -> None:
        """DV01 maior para títulos com duration maior (mantendo PU similar)."""
        dv01_ltn = self.engine.dv01(self.ltn)
        titulo_curto = TituloRendaFixa(
            nome="LTN Curto",
            tipo=TipoTitulo.PREFIXADO,
            valor_nominal=1000.0,
            taxa_cupom=0.0,
            taxa_mercado=0.12,
            vencimento=self.hoje + timedelta(days=126),
        )
        dv01_curto = self.engine.dv01(titulo_curto)
        self.assertGreater(dv01_ltn, dv01_curto)

    # =========================================================================
    # Testes de Marcação a Mercado
    # =========================================================================

    def test_mtm_retorna_resultado(self) -> None:
        """MtM deve retornar ResultadoMtM válido."""
        resultado = self.engine.marcacao_a_mercado(self.ltn, taxa_mercado=0.13)
        self.assertIsInstance(resultado, ResultadoMtM)
        self.assertGreater(resultado.pu_curva, 0)
        self.assertGreater(resultado.pu_mercado, 0)

    def test_mtm_taxa_maior_deprecia(self) -> None:
        """Quando taxa de mercado sobe, PU de mercado < PU de curva."""
        resultado = self.engine.marcacao_a_mercado(self.ltn, taxa_mercado=0.15)
        self.assertLess(resultado.pu_mercado, resultado.pu_curva)

    def test_mtm_taxa_menor_aprecia(self) -> None:
        """Quando taxa de mercado cai, PU de mercado > PU de curva."""
        resultado = self.engine.marcacao_a_mercado(self.ltn, taxa_mercado=0.08)
        self.assertGreater(resultado.pu_mercado, resultado.pu_curva)

    def test_mtm_mesma_taxa_igual(self) -> None:
        """Quando taxa de mercado = taxa original, PU curva ≈ PU mercado."""
        resultado = self.engine.marcacao_a_mercado(
            self.ltn, taxa_mercado=self.ltn.taxa_mercado
        )
        self.assertAlmostEqual(
            resultado.pu_curva, resultado.pu_mercado, delta=0.01
        )

    # =========================================================================
    # Testes de Carteira
    # =========================================================================

    def test_marcacao_carteira(self) -> None:
        """Marcação de carteira deve retornar DataFrame com todos os títulos."""
        carteira = [self.ltn, self.ntnb, self.debenture]
        taxas = {
            self.ltn.nome: 0.13,
            self.ntnb.nome: 0.06,
            self.debenture.nome: 0.14,
        }
        df = self.engine.marcacao_carteira(carteira, taxas)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("pu_mercado", df.columns)
        self.assertIn("resultado_mtm", df.columns)

    # =========================================================================
    # Testes de Cenários
    # =========================================================================

    def test_cenarios_taxa(self) -> None:
        """Cenários de taxa devem gerar DataFrame com variações."""
        df = self.engine.cenarios_taxa(
            self.ltn,
            choques_bps=[-100, -50, 0, 50, 100],
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        # PU no choque 0 deve ser o PU calculado
        pu_base = df[df["choque_bps"] == 0]["pu"].values[0]
        self.assertGreater(pu_base, 0)

    def test_estresse_carteira(self) -> None:
        """Estresse de carteira deve retornar impacto estimado."""
        carteira = [self.ltn, self.ntnb]
        resultado = self.engine.estresse_carteira(carteira, choque_bps=200)
        self.assertIsInstance(resultado, dict)
        self.assertIn("impacto_total", resultado)

    # =========================================================================
    # Testes de Spread (Z-spread e crédito)
    # =========================================================================

    def test_spread_credito(self) -> None:
        """Spread de crédito deve ser positivo quando yield > curva base."""
        spread = self.engine.calcular_spread_credito(
            yield_titulo=0.14, yield_curva_base=0.12
        )
        self.assertGreater(spread, 0)
        self.assertAlmostEqual(spread, 0.02, places=4)

    # =========================================================================
    # Testes de Relatório
    # =========================================================================

    def test_relatorio_titulo(self) -> None:
        """Relatório deve conter todas as métricas principais."""
        relatorio = self.engine.relatorio_titulo(self.ltn)
        self.assertIsInstance(relatorio, dict)
        campos_obrigatorios = ["titulo", "pu", "duration_macaulay", "dv01"]
        for campo in campos_obrigatorios:
            self.assertIn(campo, relatorio, f"Campo '{campo}' ausente no relatório")


if __name__ == "__main__":
    unittest.main()
