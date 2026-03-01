"""
QuantNucleo — Renda Fixa: Títulos e Precificação
====================================================
Implementa precificação, duration, convexidade e análise de títulos
de renda fixa brasileiros (Tesouro Direto / Debêntures / CDB / LCI/LCA).

Marcação a Mercado (MtM):
  - Compara PU (Preço Unitário) da curva vs PU de mercado
  - Calcula lucro/prejuízo marcado a mercado
  - Suporta títulos pré-fixados, IPCA+ e pós-fixados

Valuation:
  - Duration de Macaulay e Modificada
  - Convexidade
  - Key Rate Duration
  - DV01 (Dollar Value of 01)
  - Spread de crédito
  - Z-spread
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from libs.core.logging import get_logger

logger = get_logger("fixed_income.titulos")


# =============================================================================
# Enums de Renda Fixa
# =============================================================================

class TipoTitulo(str, Enum):
    """Tipo de título de renda fixa."""
    PREFIXADO = "PREFIXADO"          # LTN, NTN-F
    IPCA_PLUS = "IPCA+"              # NTN-B, NTN-B Principal
    SELIC = "SELIC"                   # LFT (Tesouro Selic)
    CDI = "CDI"                       # CDB, LCI, LCA
    CDI_MAIS = "CDI+"                 # CDI + spread
    DEBENTURE = "DEBENTURE"           # Debêntures


class IndexadorRF(str, Enum):
    """Indexador de referência."""
    PREFIXADO = "PRE"
    IPCA = "IPCA"
    SELIC = "SELIC"
    CDI = "CDI"
    IGPM = "IGPM"


class RatingCredito(str, Enum):
    """Rating de crédito simplificado (escala nacional)."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    GOVERNO = "GOV"  # Tesouro Nacional = risk-free


# =============================================================================
# Dataclasses de Renda Fixa
# =============================================================================

@dataclass
class TituloRendaFixa:
    """Representação de um título de renda fixa."""
    nome: str
    tipo: TipoTitulo
    indexador: IndexadorRF
    taxa_contratada: float        # % a.a. (ex: 0.12 = 12% a.a.)
    data_emissao: date
    data_vencimento: date
    valor_nominal: float = 1000.0  # PU de face
    cupom_semestral: float = 0.0   # % a.a. do cupom (0 se zero-coupon)
    rating: RatingCredito = RatingCredito.GOVERNO
    quantidade: float = 1.0
    preco_compra: Optional[float] = None  # PU de aquisição

    @property
    def prazo_dias_uteis(self) -> int:
        """Prazo em dias úteis (estimativa: 252 du/ano)."""
        dias_corridos = (self.data_vencimento - date.today()).days
        return max(1, int(dias_corridos * 252 / 365))

    @property
    def prazo_anos(self) -> float:
        """Prazo em anos (base 252 du)."""
        return self.prazo_dias_uteis / 252

    @property
    def tem_cupom(self) -> bool:
        return self.cupom_semestral > 0.0


@dataclass
class FluxoCaixa:
    """Fluxo de caixa de um título."""
    data: date
    valor: float
    dias_uteis: int
    fator_desconto: float = 1.0


@dataclass
class ResultadoMtM:
    """Resultado da marcação a mercado."""
    titulo: str
    data_referencia: date
    pu_curva: float           # PU pela taxa contratada (curva)
    pu_mercado: float         # PU pela taxa de mercado atual
    ganho_perda: float        # PU mercado - PU curva
    ganho_perda_pct: float    # % sobre PU curva
    ganho_perda_brl: float    # Em R$ (considerando quantidade)
    taxa_contratada: float
    taxa_mercado: float
    duration_mod: float
    dv01: float


@dataclass
class AnaliseCredito:
    """Resultado de análise de crédito."""
    spread_credito: float     # bps sobre a curva risk-free
    z_spread: float           # bps interpolado na curva
    probabilidade_default: float
    perda_esperada: float     # LGD × PD


# =============================================================================
# Motor de Precificação Renda Fixa
# =============================================================================

class RendaFixaEngine:
    """
    Motor de precificação e análise de títulos de renda fixa.

    Funcionalidades:
      - PU (Preço Unitário) via fluxo de caixa descontado
      - Duration de Macaulay e Modificada
      - Convexidade
      - DV01 (sensibilidade a 1bp de taxa)
      - Marcação a Mercado (MtM)
      - Spread de crédito e Z-spread
      - Análise de carteira RF

    Uso:
        engine = RendaFixaEngine()
        titulo = TituloRendaFixa(
            nome="Tesouro IPCA+ 2035",
            tipo=TipoTitulo.IPCA_PLUS,
            indexador=IndexadorRF.IPCA,
            taxa_contratada=0.06,
            data_emissao=date(2024, 1, 1),
            data_vencimento=date(2035, 5, 15),
            cupom_semestral=0.06,
        )
        pu = engine.calcular_pu(titulo, taxa=0.065)
        mtm = engine.marcacao_a_mercado(titulo, taxa_mercado=0.065)
    """

    DU_ANO: int = 252
    DC_ANO: int = 365

    # =========================================================================
    # Fluxo de Caixa
    # =========================================================================

    def gerar_fluxos(
        self,
        titulo: TituloRendaFixa,
        data_ref: Optional[date] = None,
    ) -> list[FluxoCaixa]:
        """
        Gera fluxos de caixa futuros do título.

        Para títulos com cupom semestral (NTN-F, NTN-B):
          - Cupom a cada 6 meses
          - Amortização + último cupom no vencimento

        Para zero-coupon (LTN, NTN-B Principal, LFT):
          - Apenas pagamento no vencimento
        """
        ref = data_ref or date.today()
        fluxos: list[FluxoCaixa] = []

        if titulo.tem_cupom:
            # Taxa do cupom semestral
            cupom_semestral_valor = (
                (1 + titulo.cupom_semestral) ** 0.5 - 1
            ) * titulo.valor_nominal

            dt = titulo.data_vencimento
            datas_cupom: list[date] = []

            # Gera datas de cupom retroativamente a partir do vencimento
            while dt > ref:
                datas_cupom.append(dt)
                # Retrocede ~6 meses
                if dt.month > 6:
                    dt = dt.replace(month=dt.month - 6)
                else:
                    dt = dt.replace(year=dt.year - 1, month=dt.month + 6)

            datas_cupom.sort()

            for i, data_cupom in enumerate(datas_cupom):
                dias = max(1, self._dias_uteis_entre(ref, data_cupom))
                if data_cupom == titulo.data_vencimento:
                    valor = titulo.valor_nominal + cupom_semestral_valor
                else:
                    valor = cupom_semestral_valor

                fluxos.append(FluxoCaixa(
                    data=data_cupom,
                    valor=valor,
                    dias_uteis=dias,
                ))
        else:
            # Zero-coupon: só paga no vencimento
            dias = max(1, self._dias_uteis_entre(ref, titulo.data_vencimento))
            fluxos.append(FluxoCaixa(
                data=titulo.data_vencimento,
                valor=titulo.valor_nominal,
                dias_uteis=dias,
            ))

        return fluxos

    # =========================================================================
    # Precificação — PU
    # =========================================================================

    def calcular_pu(
        self,
        titulo: TituloRendaFixa,
        taxa: Optional[float] = None,
        data_ref: Optional[date] = None,
    ) -> float:
        """
        Calcula o Preço Unitário (PU) via DCF.

        PU = Σ [ FC_i / (1 + taxa)^(du_i / 252) ]

        Args:
            titulo: Título a precificar
            taxa: Taxa de desconto a.a. (None = usa taxa contratada)
            data_ref: Data de referência (None = hoje)

        Returns:
            PU (preço unitário)
        """
        taxa_desc = taxa if taxa is not None else titulo.taxa_contratada
        fluxos = self.gerar_fluxos(titulo, data_ref)

        if not fluxos:
            return titulo.valor_nominal

        pu = 0.0
        for fc in fluxos:
            fator = (1 + taxa_desc) ** (fc.dias_uteis / self.DU_ANO)
            fc.fator_desconto = fator
            pu += fc.valor / fator

        return float(pu)

    def calcular_ytm(
        self,
        titulo: TituloRendaFixa,
        pu_mercado: float,
        data_ref: Optional[date] = None,
    ) -> float:
        """
        Calcula o Yield to Maturity (YTM) dado um PU de mercado.

        Encontra a taxa `y` tal que:
            PU_mercado = Σ [ FC_i / (1 + y)^(du_i / 252) ]

        Usa Brent root-finding.
        """
        def f(y: float) -> float:
            return self.calcular_pu(titulo, taxa=y, data_ref=data_ref) - pu_mercado

        try:
            ytm = brentq(f, -0.5, 2.0, xtol=1e-10, maxiter=500)
            return float(ytm)
        except Exception as e:
            logger.warning("ytm_calculo_falhou", erro=str(e), pu=pu_mercado)
            return titulo.taxa_contratada

    # =========================================================================
    # Duration & Convexidade
    # =========================================================================

    def duration_macaulay(
        self,
        titulo: TituloRendaFixa,
        taxa: Optional[float] = None,
        data_ref: Optional[date] = None,
    ) -> float:
        """
        Duration de Macaulay (em anos).

        D_mac = Σ [ t_i × VP(FC_i) ] / PU

        Onde t_i = du_i / 252 (tempo em anos até fluxo i).
        """
        taxa_desc = taxa if taxa is not None else titulo.taxa_contratada
        fluxos = self.gerar_fluxos(titulo, data_ref)
        pu = self.calcular_pu(titulo, taxa_desc, data_ref)

        if pu == 0 or not fluxos:
            return 0.0

        soma_ponderada = 0.0
        for fc in fluxos:
            t = fc.dias_uteis / self.DU_ANO
            fator = (1 + taxa_desc) ** t
            vp = fc.valor / fator
            soma_ponderada += t * vp

        return float(soma_ponderada / pu)

    def duration_modificada(
        self,
        titulo: TituloRendaFixa,
        taxa: Optional[float] = None,
        data_ref: Optional[date] = None,
    ) -> float:
        """
        Duration Modificada — sensibilidade do PU a variações de taxa.

        D_mod = D_mac / (1 + y)

        Interpretação: se a taxa subir 1%, o PU cai ~D_mod%.
        """
        taxa_desc = taxa if taxa is not None else titulo.taxa_contratada
        d_mac = self.duration_macaulay(titulo, taxa_desc, data_ref)
        return float(d_mac / (1 + taxa_desc))

    def convexidade(
        self,
        titulo: TituloRendaFixa,
        taxa: Optional[float] = None,
        data_ref: Optional[date] = None,
    ) -> float:
        """
        Convexidade — curvatura da relação preço/yield.

        C = (1/PU) × Σ [ t_i × (t_i + 1) × VP(FC_i) / (1 + y)^2 ]

        Usada para melhorar a estimativa de ΔPU quando taxas variam muito.
        """
        taxa_desc = taxa if taxa is not None else titulo.taxa_contratada
        fluxos = self.gerar_fluxos(titulo, data_ref)
        pu = self.calcular_pu(titulo, taxa_desc, data_ref)

        if pu == 0 or not fluxos:
            return 0.0

        soma = 0.0
        for fc in fluxos:
            t = fc.dias_uteis / self.DU_ANO
            fator = (1 + taxa_desc) ** t
            vp = fc.valor / fator
            soma += t * (t + 1) * vp

        return float(soma / (pu * (1 + taxa_desc) ** 2))

    def dv01(
        self,
        titulo: TituloRendaFixa,
        taxa: Optional[float] = None,
        data_ref: Optional[date] = None,
    ) -> float:
        """
        DV01 — Dollar Value of 01 (variação de PU para +1bp de taxa).

        DV01 ≈ PU × D_mod × 0.0001

        Uso: quanto perde/ganha em R$ por 1bp de movimento na curva.
        """
        taxa_desc = taxa if taxa is not None else titulo.taxa_contratada
        pu = self.calcular_pu(titulo, taxa_desc, data_ref)
        d_mod = self.duration_modificada(titulo, taxa_desc, data_ref)
        return float(pu * d_mod * 0.0001)

    def key_rate_duration(
        self,
        titulo: TituloRendaFixa,
        vertices: list[float],
        data_ref: Optional[date] = None,
        bump: float = 0.0001,
    ) -> dict[float, float]:
        """
        Key Rate Duration — sensibilidade do PU a cada vértice da curva.

        Perturba a taxa em cada vértice (em anos) em +1bp e mede o impacto.
        Útil para gestão de duration em múltiplos pontos da curva.

        Args:
            titulo: Título a analisar
            vertices: Lista de vértices em anos (ex: [0.5, 1, 2, 5, 10])
            bump: Tamanho do choque em decimal (0.0001 = 1bp)

        Returns:
            dict vértice → KRD
        """
        taxa_base = titulo.taxa_contratada
        pu_base = self.calcular_pu(titulo, taxa_base, data_ref)
        krd: dict[float, float] = {}

        for v in vertices:
            # Bump só nos fluxos próximos ao vértice
            pu_up = self._pu_com_bump_vertice(titulo, v, bump, data_ref)
            pu_down = self._pu_com_bump_vertice(titulo, v, -bump, data_ref)
            krd[v] = float(-(pu_up - pu_down) / (2 * bump * pu_base))

        return krd

    def _pu_com_bump_vertice(
        self,
        titulo: TituloRendaFixa,
        vertice_anos: float,
        bump: float,
        data_ref: Optional[date] = None,
    ) -> float:
        """Calcula PU com bump na taxa aplicado proporcionalmente ao vértice."""
        taxa_base = titulo.taxa_contratada
        fluxos = self.gerar_fluxos(titulo, data_ref)

        pu = 0.0
        for fc in fluxos:
            t = fc.dias_uteis / self.DU_ANO
            # Peso do bump: maior perto do vértice, decai linearmente
            peso_bump = max(0.0, 1.0 - abs(t - vertice_anos) / max(vertice_anos, 0.5))
            taxa_ajustada = taxa_base + bump * peso_bump
            fator = (1 + taxa_ajustada) ** t
            pu += fc.valor / fator

        return float(pu)

    # =========================================================================
    # Marcação a Mercado (MtM)
    # =========================================================================

    def marcacao_a_mercado(
        self,
        titulo: TituloRendaFixa,
        taxa_mercado: float,
        data_ref: Optional[date] = None,
    ) -> ResultadoMtM:
        """
        Marcação a Mercado — compara PU curva vs PU mercado.

        Conceito:
          - PU Curva: preço calculado pela taxa original contratada
          - PU Mercado: preço calculado pela taxa negociada hoje
          - Se taxa caiu → PU mercado > PU curva → LUCRO MtM
          - Se taxa subiu → PU mercado < PU curva → PREJUÍZO MtM

        Args:
            titulo: Título a marcar
            taxa_mercado: Taxa atual de mercado (a.a.)
            data_ref: Data de referência

        Returns:
            ResultadoMtM com detalhamento completo
        """
        ref = data_ref or date.today()

        pu_curva = self.calcular_pu(titulo, titulo.taxa_contratada, ref)
        pu_mercado = self.calcular_pu(titulo, taxa_mercado, ref)

        ganho_perda = pu_mercado - pu_curva
        ganho_perda_pct = ganho_perda / pu_curva if pu_curva != 0 else 0.0
        ganho_perda_brl = ganho_perda * titulo.quantidade

        d_mod = self.duration_modificada(titulo, taxa_mercado, ref)
        dv01_val = self.dv01(titulo, taxa_mercado, ref)

        resultado = ResultadoMtM(
            titulo=titulo.nome,
            data_referencia=ref,
            pu_curva=round(pu_curva, 6),
            pu_mercado=round(pu_mercado, 6),
            ganho_perda=round(ganho_perda, 6),
            ganho_perda_pct=round(ganho_perda_pct, 6),
            ganho_perda_brl=round(ganho_perda_brl, 2),
            taxa_contratada=titulo.taxa_contratada,
            taxa_mercado=taxa_mercado,
            duration_mod=round(d_mod, 4),
            dv01=round(dv01_val, 6),
        )

        emoji = "📈" if ganho_perda >= 0 else "📉"
        logger.info(
            "marcacao_a_mercado",
            titulo=titulo.nome,
            pu_curva=pu_curva,
            pu_mercado=pu_mercado,
            ganho_perda_brl=ganho_perda_brl,
            direcao=emoji,
        )

        return resultado

    def marcacao_carteira(
        self,
        titulos: list[TituloRendaFixa],
        taxas_mercado: dict[str, float],
        data_ref: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Marcação a mercado de uma carteira inteira de renda fixa.

        Args:
            titulos: Lista de títulos
            taxas_mercado: Dict nome_titulo → taxa atual de mercado
            data_ref: Data de referência

        Returns:
            DataFrame com PU curva, PU mercado, ganho/perda por título
        """
        resultados: list[dict] = []

        for titulo in titulos:
            taxa_mk = taxas_mercado.get(titulo.nome, titulo.taxa_contratada)
            mtm = self.marcacao_a_mercado(titulo, taxa_mk, data_ref)

            resultados.append({
                "titulo": mtm.titulo,
                "tipo": titulo.tipo.value,
                "indexador": titulo.indexador.value,
                "rating": titulo.rating.value,
                "vencimento": titulo.data_vencimento,
                "taxa_contratada": mtm.taxa_contratada,
                "taxa_mercado": mtm.taxa_mercado,
                "pu_curva": mtm.pu_curva,
                "pu_mercado": mtm.pu_mercado,
                "quantidade": titulo.quantidade,
                "ganho_perda_pct": mtm.ganho_perda_pct,
                "ganho_perda_brl": mtm.ganho_perda_brl,
                "duration_mod": mtm.duration_mod,
                "dv01": mtm.dv01,
            })

        df = pd.DataFrame(resultados)

        if not df.empty:
            total_brl = df["ganho_perda_brl"].sum()
            emoji = "📈" if total_brl >= 0 else "📉"
            logger.info(
                "marcacao_carteira_completa",
                total_titulos=len(titulos),
                ganho_perda_total_brl=round(total_brl, 2),
                direcao=emoji,
            )

        return df

    # =========================================================================
    # Spread de Crédito
    # =========================================================================

    def calcular_spread_credito(
        self,
        titulo: TituloRendaFixa,
        pu_mercado: float,
        taxa_livre_risco: float,
        data_ref: Optional[date] = None,
    ) -> float:
        """
        Calcula spread de crédito sobre a taxa livre de risco.

        Spread = YTM do título - taxa risk-free

        Útil para avaliar prêmio de risco de debêntures e CDBs.

        Args:
            titulo: Título corporativo
            pu_mercado: PU observado no mercado
            taxa_livre_risco: Taxa do Tesouro equivalente (a.a.)

        Returns:
            Spread em decimal (ex: 0.018 = 180bps)
        """
        ytm = self.calcular_ytm(titulo, pu_mercado, data_ref)
        spread = ytm - taxa_livre_risco

        logger.info(
            "spread_credito",
            titulo=titulo.nome,
            ytm=round(ytm, 6),
            rf=round(taxa_livre_risco, 6),
            spread_bps=round(spread * 10000, 1),
        )

        return float(spread)

    def z_spread(
        self,
        titulo: TituloRendaFixa,
        pu_mercado: float,
        curva_juros: dict[float, float],
        data_ref: Optional[date] = None,
    ) -> float:
        """
        Z-Spread — spread constante adicionado a cada ponto da curva de juros.

        Encontra z tal que:
            PU_mercado = Σ [ FC_i / (1 + r_i + z)^(t_i) ]

        Onde r_i é a taxa spot interpolada na curva para o prazo t_i.

        Args:
            titulo: Título a analisar
            pu_mercado: PU observado no mercado
            curva_juros: Dict vértice_anos → taxa spot (ex: {0.5: 0.10, 1: 0.105})

        Returns:
            Z-spread em decimal
        """
        fluxos = self.gerar_fluxos(titulo, data_ref)

        # Interpolar curva
        vertices = sorted(curva_juros.keys())
        taxas = [curva_juros[v] for v in vertices]

        def pu_com_zspread(z: float) -> float:
            pu = 0.0
            for fc in fluxos:
                t = fc.dias_uteis / self.DU_ANO
                # Interpolação linear da taxa spot
                taxa_spot = float(np.interp(t, vertices, taxas))
                fator = (1 + taxa_spot + z) ** t
                pu += fc.valor / fator
            return pu

        def f(z: float) -> float:
            return pu_com_zspread(z) - pu_mercado

        try:
            zs = brentq(f, -0.5, 2.0, xtol=1e-10, maxiter=500)
            logger.info(
                "z_spread_calculado",
                titulo=titulo.nome,
                z_spread_bps=round(zs * 10000, 1),
            )
            return float(zs)
        except Exception as e:
            logger.warning("z_spread_falhou", erro=str(e))
            return 0.0

    # =========================================================================
    # Análise de Sensibilidade
    # =========================================================================

    def cenarios_taxa(
        self,
        titulo: TituloRendaFixa,
        variacao_bps: list[int],
        data_ref: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Simula cenários de variação de taxa e impacto no PU.

        Gera tabela com: variação (bps), taxa resultante, PU, ΔPU, ΔPU%.

        Args:
            titulo: Título a analisar
            variacao_bps: Lista de variações em basis points (ex: [-100, -50, 0, 50, 100])

        Returns:
            DataFrame com cenários
        """
        taxa_base = titulo.taxa_contratada
        pu_base = self.calcular_pu(titulo, taxa_base, data_ref)

        cenarios: list[dict] = []
        for bp in variacao_bps:
            taxa_cenario = taxa_base + bp / 10000
            pu_cenario = self.calcular_pu(titulo, taxa_cenario, data_ref)
            delta_pu = pu_cenario - pu_base

            cenarios.append({
                "variacao_bps": bp,
                "taxa": taxa_cenario,
                "pu": round(pu_cenario, 6),
                "delta_pu": round(delta_pu, 6),
                "delta_pu_pct": round(delta_pu / pu_base * 100, 4) if pu_base != 0 else 0,
                "delta_brl": round(delta_pu * titulo.quantidade, 2),
            })

        return pd.DataFrame(cenarios)

    def estresse_carteira(
        self,
        titulos: list[TituloRendaFixa],
        choque_bps: int,
        data_ref: Optional[date] = None,
    ) -> dict:
        """
        Teste de estresse: aplica choque paralelo de taxa na carteira.

        Args:
            titulos: Carteira de RF
            choque_bps: Choque em basis points (ex: +200)

        Returns:
            Dict com perda total, por título, e durations
        """
        total_perda = 0.0
        detalhes: list[dict] = []

        for titulo in titulos:
            pu_base = self.calcular_pu(titulo, titulo.taxa_contratada, data_ref)
            taxa_estresse = titulo.taxa_contratada + choque_bps / 10000
            pu_estresse = self.calcular_pu(titulo, taxa_estresse, data_ref)
            perda = (pu_estresse - pu_base) * titulo.quantidade

            total_perda += perda
            detalhes.append({
                "titulo": titulo.nome,
                "pu_base": round(pu_base, 4),
                "pu_estresse": round(pu_estresse, 4),
                "perda_brl": round(perda, 2),
                "duration_mod": round(
                    self.duration_modificada(titulo, titulo.taxa_contratada, data_ref), 4
                ),
            })

        logger.info(
            "estresse_carteira",
            choque_bps=choque_bps,
            perda_total=round(total_perda, 2),
        )

        return {
            "choque_bps": choque_bps,
            "perda_total_brl": round(total_perda, 2),
            "detalhes": detalhes,
        }

    # =========================================================================
    # Relatório Consolidado
    # =========================================================================

    def relatorio_titulo(
        self,
        titulo: TituloRendaFixa,
        taxa_mercado: Optional[float] = None,
        data_ref: Optional[date] = None,
    ) -> dict:
        """
        Gera relatório completo de um título de renda fixa.

        Inclui: PU, Duration, Convexidade, DV01, MtM, cenários.
        """
        taxa = taxa_mercado or titulo.taxa_contratada
        ref = data_ref or date.today()

        pu = self.calcular_pu(titulo, taxa, ref)
        d_mac = self.duration_macaulay(titulo, taxa, ref)
        d_mod = self.duration_modificada(titulo, taxa, ref)
        conv = self.convexidade(titulo, taxa, ref)
        dv01_val = self.dv01(titulo, taxa, ref)

        relatorio: dict = {
            "titulo": titulo.nome,
            "tipo": titulo.tipo.value,
            "indexador": titulo.indexador.value,
            "rating": titulo.rating.value,
            "vencimento": str(titulo.data_vencimento),
            "prazo_anos": round(titulo.prazo_anos, 2),
            "taxa_contratada": titulo.taxa_contratada,
            "pu": round(pu, 6),
            "duration_macaulay": round(d_mac, 4),
            "duration_modificada": round(d_mod, 4),
            "convexidade": round(conv, 4),
            "dv01": round(dv01_val, 6),
        }

        # MtM se taxa de mercado for diferente
        if taxa_mercado is not None and taxa_mercado != titulo.taxa_contratada:
            mtm = self.marcacao_a_mercado(titulo, taxa_mercado, ref)
            relatorio["mtm"] = {
                "pu_curva": mtm.pu_curva,
                "pu_mercado": mtm.pu_mercado,
                "ganho_perda_pct": mtm.ganho_perda_pct,
                "ganho_perda_brl": mtm.ganho_perda_brl,
            }

        # Cenários
        cenarios = self.cenarios_taxa(
            titulo,
            [-200, -100, -50, -25, 0, 25, 50, 100, 200],
            ref,
        )
        relatorio["cenarios"] = cenarios.to_dict("records")

        logger.info("relatorio_titulo_gerado", titulo=titulo.nome)
        return relatorio

    # =========================================================================
    # Utilitários
    # =========================================================================

    @staticmethod
    def _dias_uteis_entre(data_inicio: date, data_fim: date) -> int:
        """Estima dias úteis entre duas datas (252 du/365 dc)."""
        dias_corridos = (data_fim - data_inicio).days
        return max(1, int(dias_corridos * 252 / 365))
