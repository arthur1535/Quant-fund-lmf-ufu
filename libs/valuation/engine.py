"""
QuantNucleo — Motor de Valuation de Ações
===========================================
Implementa múltiplos métodos de valuation para análise fundamentalista
e seleção de ativos (ações BR e US).

Métodos implementados:
 - DCF (Discounted Cash Flow) — FCFF e FCFE
 - DDM (Dividend Discount Model) — Gordon Growth
 - Múltiplos (P/E, P/B, EV/EBITDA, EV/FCF, P/FCF)
 - Earnings Yield, FCF Yield, Dividend Yield
 - Piotroski F-Score (força fundamentalista)
 - Altman Z-Score (risco de falência)
 - Graham Number
 - Relative Valuation (ranking por múltiplos)
 - WACC (Weighted Average Cost of Capital)
 - Fair Value composite (média ponderada dos métodos)

Reaproveitado e estendido de:
 - eua/analiseempresasamericanas.py: métricas base de valuation
 - Novas implementações: DCF, Piotroski, Altman, Graham
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

from libs.core.logging import get_logger

logger = get_logger("valuation.engine")


# =============================================================================
# Dataclasses de Valuation
# =============================================================================

@dataclass
class DadosFundamentalistas:
    """Dados fundamentalistas de um ativo (snapshot)."""
    ticker: str
    data_referencia: date

    # Preço & Mercado
    preco_atual: float = 0.0
    market_cap: float = 0.0
    shares_outstanding: float = 0.0
    enterprise_value: float = 0.0

    # Demonstração de Resultados
    receita_liquida: float = 0.0
    ebitda: float = 0.0
    ebit: float = 0.0
    lucro_liquido: float = 0.0
    lpa: float = 0.0  # Lucro por ação (EPS)

    # Balanço Patrimonial
    ativo_total: float = 0.0
    ativo_circulante: float = 0.0
    passivo_total: float = 0.0
    passivo_circulante: float = 0.0
    patrimonio_liquido: float = 0.0
    divida_bruta: float = 0.0
    divida_liquida: float = 0.0
    caixa: float = 0.0
    capital_giro: float = 0.0

    # Fluxo de Caixa
    fluxo_caixa_operacional: float = 0.0
    capex: float = 0.0
    fcf: float = 0.0  # Free Cash Flow (FCO - CAPEX)
    dividendos_pagos: float = 0.0

    # Margens
    margem_bruta: float = 0.0
    margem_ebitda: float = 0.0
    margem_liquida: float = 0.0
    margem_fcf: float = 0.0

    # Retornos
    roe: float = 0.0         # Return on Equity
    roa: float = 0.0         # Return on Assets
    roic: float = 0.0        # Return on Invested Capital

    # Dividendos
    dividendo_por_acao: float = 0.0
    dividend_yield: float = 0.0
    payout_ratio: float = 0.0

    # Metadados
    setor: str = ""
    moeda: str = "BRL"

    @property
    def fcf_por_acao(self) -> float:
        if self.shares_outstanding == 0:
            return 0.0
        return self.fcf / self.shares_outstanding

    @property
    def valor_patrimonial_por_acao(self) -> float:
        if self.shares_outstanding == 0:
            return 0.0
        return self.patrimonio_liquido / self.shares_outstanding


@dataclass
class ResultadoValuation:
    """Resultado consolidado de valuation."""
    ticker: str
    data_referencia: date
    preco_atual: float

    # Fair values por método
    dcf_fair_value: Optional[float] = None
    ddm_fair_value: Optional[float] = None
    graham_number: Optional[float] = None
    multiplos_fair_value: Optional[float] = None

    # Composite
    fair_value_composto: Optional[float] = None
    upside_downside: Optional[float] = None  # % relativo ao preço atual

    # Scores
    piotroski_f_score: Optional[int] = None
    altman_z_score: Optional[float] = None

    # Múltiplos
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_fcf: Optional[float] = None
    earnings_yield: Optional[float] = None
    fcf_yield: Optional[float] = None
    dividend_yield: Optional[float] = None

    # Rating qualitativo
    recomendacao: str = ""  # "SUBVALORIZADO", "JUSTO", "SOBREVALORIZADO"
    confianca: float = 0.0  # 0 a 1


# =============================================================================
# Motor de Valuation
# =============================================================================

class ValuationEngine:
    """
    Motor de valuation multi-método para ações.

    Uso:
        engine = ValuationEngine()
        dados = engine.coletar_dados_yfinance("PETR4.SA")
        resultado = engine.valuation_completo(dados)
    """

    # =========================================================================
    # Coleta de Dados via yfinance
    # =========================================================================

    def coletar_dados_yfinance(self, ticker: str) -> DadosFundamentalistas:
        """
        Coleta dados fundamentalistas de um ticker via yfinance.

        Compatível com ações BR (.SA) e US.
        """
        import yfinance as yf

        logger.info("coletando_dados_fundamentalistas", ticker=ticker)
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        bs = tk.balance_sheet
        cf = tk.cash_flow
        fin = tk.financials

        preco = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        shares = info.get("sharesOutstanding", 0)

        # Balanço (último período disponível)
        ativo_total = self._ultimo_valor(bs, "Total Assets")
        ativo_circ = self._ultimo_valor(bs, "Current Assets")
        passivo_total = self._ultimo_valor(bs, "Total Liabilities Net Minority Interest")
        passivo_circ = self._ultimo_valor(bs, "Current Liabilities")
        pl = self._ultimo_valor(bs, "Stockholders Equity")
        divida_bruta = self._ultimo_valor(bs, "Total Debt")
        caixa = self._ultimo_valor(bs, "Cash And Cash Equivalents")

        # DRE
        receita = self._ultimo_valor(fin, "Total Revenue")
        ebitda_val = info.get("ebitda", 0) or 0
        ebit_val = self._ultimo_valor(fin, "EBIT")
        lucro = self._ultimo_valor(fin, "Net Income")

        # Fluxo de Caixa
        fco = self._ultimo_valor(cf, "Operating Cash Flow")
        capex_val = abs(self._ultimo_valor(cf, "Capital Expenditure"))
        dividendos = abs(self._ultimo_valor(cf, "Common Stock Dividend Paid"))

        fcf_val = fco - capex_val
        divida_liq = divida_bruta - caixa

        # Margens
        margem_bruta = info.get("grossMargins", 0) or 0
        margem_liquida = (lucro / receita) if receita != 0 else 0
        margem_ebitda = (ebitda_val / receita) if receita != 0 else 0
        margem_fcf = (fcf_val / receita) if receita != 0 else 0

        # Retornos
        roe = info.get("returnOnEquity", 0) or 0
        roa = info.get("returnOnAssets", 0) or 0
        roic_val = (ebit_val * (1 - 0.34)) / (pl + divida_liq) if (pl + divida_liq) != 0 else 0

        dados = DadosFundamentalistas(
            ticker=ticker,
            data_referencia=date.today(),
            preco_atual=preco,
            market_cap=info.get("marketCap", 0) or 0,
            shares_outstanding=shares,
            enterprise_value=info.get("enterpriseValue", 0) or 0,
            receita_liquida=receita,
            ebitda=ebitda_val,
            ebit=ebit_val,
            lucro_liquido=lucro,
            lpa=info.get("trailingEps", 0) or 0,
            ativo_total=ativo_total,
            ativo_circulante=ativo_circ,
            passivo_total=passivo_total,
            passivo_circulante=passivo_circ,
            patrimonio_liquido=pl,
            divida_bruta=divida_bruta,
            divida_liquida=divida_liq,
            caixa=caixa,
            capital_giro=ativo_circ - passivo_circ,
            fluxo_caixa_operacional=fco,
            capex=capex_val,
            fcf=fcf_val,
            dividendos_pagos=dividendos,
            margem_bruta=margem_bruta,
            margem_ebitda=margem_ebitda,
            margem_liquida=margem_liquida,
            margem_fcf=margem_fcf,
            roe=roe,
            roa=roa,
            roic=roic_val,
            dividendo_por_acao=info.get("dividendRate", 0) or 0,
            dividend_yield=info.get("dividendYield", 0) or 0,
            payout_ratio=info.get("payoutRatio", 0) or 0,
            setor=info.get("sector", ""),
            moeda="BRL" if ticker.endswith(".SA") else "USD",
        )

        logger.info(
            "dados_fundamentalistas_coletados",
            ticker=ticker,
            preco=preco,
            market_cap=dados.market_cap,
            pe=round(preco / dados.lpa, 2) if dados.lpa != 0 else None,
        )

        return dados

    # =========================================================================
    # DCF — Discounted Cash Flow (FCFF)
    # =========================================================================

    def dcf_fcff(
        self,
        dados: DadosFundamentalistas,
        taxa_crescimento_5y: float = 0.05,
        taxa_crescimento_perpetua: float = 0.03,
        wacc: Optional[float] = None,
        anos_projecao: int = 10,
    ) -> float:
        """
        DCF via Free Cash Flow to Firm (FCFF).

        Modelo de dois estágios:
          1. Crescimento alto nos primeiros anos_projecao anos
          2. Perpetuidade (terminal value) com crescimento g

        Fair Value = (Σ FCF_t / (1+WACC)^t + TV / (1+WACC)^n - Dívida Líq.) / Ações

        Args:
            dados: Dados fundamentalistas do ativo
            taxa_crescimento_5y: Crescimento do FCF nos primeiros anos
            taxa_crescimento_perpetua: Crescimento na perpetuidade (g)
            wacc: WACC (None = calcula automaticamente)
            anos_projecao: Número de anos projetados

        Returns:
            Fair value por ação via DCF
        """
        if dados.fcf <= 0:
            logger.warning("dcf_fcf_negativo", ticker=dados.ticker, fcf=dados.fcf)
            return 0.0

        taxa_desc = wacc or self.calcular_wacc(dados)

        if taxa_desc <= taxa_crescimento_perpetua:
            logger.warning("dcf_wacc_menor_g", wacc=taxa_desc, g=taxa_crescimento_perpetua)
            taxa_desc = taxa_crescimento_perpetua + 0.02

        # Projetar FCFs
        fcfs_projetados = []
        fcf_atual = dados.fcf

        for t in range(1, anos_projecao + 1):
            # Decaimento linear da taxa de crescimento
            fator_decaimento = 1 - (t - 1) / anos_projecao
            taxa_t = (
                taxa_crescimento_5y * fator_decaimento
                + taxa_crescimento_perpetua * (1 - fator_decaimento)
            )
            fcf_atual = fcf_atual * (1 + taxa_t)
            vp = fcf_atual / (1 + taxa_desc) ** t
            fcfs_projetados.append(vp)

        # Terminal value (Gordon Growth)
        fcf_terminal = fcf_atual * (1 + taxa_crescimento_perpetua)
        terminal_value = fcf_terminal / (taxa_desc - taxa_crescimento_perpetua)
        vp_terminal = terminal_value / (1 + taxa_desc) ** anos_projecao

        # Enterprise Value → Equity Value
        enterprise_value = sum(fcfs_projetados) + vp_terminal
        equity_value = enterprise_value - dados.divida_liquida + dados.caixa

        if dados.shares_outstanding == 0:
            return 0.0

        fair_value = equity_value / dados.shares_outstanding

        logger.info(
            "dcf_calculado",
            ticker=dados.ticker,
            fair_value=round(fair_value, 2),
            ev=round(enterprise_value, 0),
            wacc=round(taxa_desc, 4),
        )

        return float(max(0, fair_value))

    # =========================================================================
    # DDM — Dividend Discount Model (Gordon Growth)
    # =========================================================================

    def ddm_gordon(
        self,
        dados: DadosFundamentalistas,
        taxa_crescimento_dividendo: float = 0.04,
        custo_capital_proprio: Optional[float] = None,
    ) -> float:
        """
        Gordon Growth Model (DDM de um estágio).

        P₀ = D₁ / (ke - g)

        Onde:
          D₁ = dividendo esperado no próximo período
          ke = custo do capital próprio
          g = taxa de crescimento perpétua dos dividendos

        Args:
            dados: Dados fundamentalistas
            taxa_crescimento_dividendo: Taxa g de crescimento
            custo_capital_proprio: ke (None = calcula via CAPM)

        Returns:
            Fair value por ação via DDM
        """
        if dados.dividendo_por_acao <= 0:
            logger.warning("ddm_sem_dividendos", ticker=dados.ticker)
            return 0.0

        ke = custo_capital_proprio or self.custo_capital_capm(dados)

        if ke <= taxa_crescimento_dividendo:
            logger.warning("ddm_ke_menor_g", ke=ke, g=taxa_crescimento_dividendo)
            return 0.0

        d1 = dados.dividendo_por_acao * (1 + taxa_crescimento_dividendo)
        fair_value = d1 / (ke - taxa_crescimento_dividendo)

        logger.info(
            "ddm_calculado",
            ticker=dados.ticker,
            fair_value=round(fair_value, 2),
            d1=round(d1, 4),
            ke=round(ke, 4),
        )

        return float(max(0, fair_value))

    def ddm_dois_estagios(
        self,
        dados: DadosFundamentalistas,
        g_alto: float = 0.08,
        g_perpetuo: float = 0.04,
        anos_alto: int = 5,
        ke: Optional[float] = None,
    ) -> float:
        """
        DDM de dois estágios:
          - Fase 1: crescimento alto (g_alto) por n anos
          - Fase 2: perpetuidade com crescimento g_perpetuo

        Mais realista que Gordon puro para empresas em crescimento.
        """
        if dados.dividendo_por_acao <= 0:
            return 0.0

        custo_eq = ke or self.custo_capital_capm(dados)
        if custo_eq <= g_perpetuo:
            return 0.0

        # Fase 1
        div = dados.dividendo_por_acao
        vp_fase1 = 0.0
        for t in range(1, anos_alto + 1):
            div = div * (1 + g_alto)
            vp_fase1 += div / (1 + custo_eq) ** t

        # Fase 2 (perpetuidade)
        div_terminal = div * (1 + g_perpetuo)
        tv = div_terminal / (custo_eq - g_perpetuo)
        vp_tv = tv / (1 + custo_eq) ** anos_alto

        return float(max(0, vp_fase1 + vp_tv))

    # =========================================================================
    # Graham Number
    # =========================================================================

    def graham_number(self, dados: DadosFundamentalistas) -> float:
        """
        Graham Number — preço justo segundo Benjamin Graham.

        Graham # = √(22.5 × EPS × BVPS)

        Onde:
          EPS = Lucro por ação (trailing)
          BVPS = Valor patrimonial por ação

        Graham sugeria: comprar quando preço < Graham Number.
        """
        eps = dados.lpa
        bvps = dados.valor_patrimonial_por_acao

        if eps <= 0 or bvps <= 0:
            logger.warning(
                "graham_dados_negativos",
                ticker=dados.ticker,
                eps=eps,
                bvps=bvps,
            )
            return 0.0

        gn = float(np.sqrt(22.5 * eps * bvps))

        logger.info("graham_number", ticker=dados.ticker, valor=round(gn, 2))
        return gn

    # =========================================================================
    # Múltiplos de Valuation
    # =========================================================================

    def calcular_multiplos(self, dados: DadosFundamentalistas) -> dict[str, Optional[float]]:
        """
        Calcula múltiplos de valuation.

        Retorna:
            P/E, P/B, EV/EBITDA, EV/FCF, P/FCF,
            Earnings Yield, FCF Yield, Dividend Yield
        """
        preco = dados.preco_atual

        pe = (preco / dados.lpa) if dados.lpa > 0 else None
        pb = (preco / dados.valor_patrimonial_por_acao) if dados.valor_patrimonial_por_acao > 0 else None
        ev_ebitda = (dados.enterprise_value / dados.ebitda) if dados.ebitda > 0 else None
        ev_fcf = (dados.enterprise_value / dados.fcf) if dados.fcf > 0 else None
        p_fcf = (dados.market_cap / dados.fcf) if dados.fcf > 0 else None

        earnings_yield = (1 / pe) if pe and pe > 0 else None
        fcf_yield = (dados.fcf / dados.market_cap) if dados.market_cap > 0 else None
        div_yield = dados.dividend_yield

        multiplos = {
            "pe_ratio": round(pe, 2) if pe else None,
            "pb_ratio": round(pb, 2) if pb else None,
            "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
            "ev_fcf": round(ev_fcf, 2) if ev_fcf else None,
            "p_fcf": round(p_fcf, 2) if p_fcf else None,
            "earnings_yield": round(earnings_yield, 4) if earnings_yield else None,
            "fcf_yield": round(fcf_yield, 4) if fcf_yield else None,
            "dividend_yield": round(div_yield, 4) if div_yield else None,
            "ev_receita": (
                round(dados.enterprise_value / dados.receita_liquida, 2)
                if dados.receita_liquida > 0 else None
            ),
            "debt_equity": (
                round(dados.divida_liquida / dados.patrimonio_liquido, 2)
                if dados.patrimonio_liquido > 0 else None
            ),
            "current_ratio": (
                round(dados.ativo_circulante / dados.passivo_circulante, 2)
                if dados.passivo_circulante > 0 else None
            ),
        }

        logger.info("multiplos_calculados", ticker=dados.ticker, pe=pe, ev_ebitda=ev_ebitda)
        return multiplos

    def fair_value_por_multiplos(
        self,
        dados: DadosFundamentalistas,
        pe_justo: float = 15.0,
        ev_ebitda_justo: float = 10.0,
    ) -> float:
        """
        Fair value por média de múltiplos-alvo.

        Calcula preço implícito por P/E e EV/EBITDA de referência do setor.
        """
        valores: list[float] = []

        # Fair value via P/E
        if dados.lpa > 0:
            fv_pe = dados.lpa * pe_justo
            valores.append(fv_pe)

        # Fair value via EV/EBITDA
        if dados.ebitda > 0 and dados.shares_outstanding > 0:
            ev_implico = dados.ebitda * ev_ebitda_justo
            eq_value = ev_implico - dados.divida_liquida + dados.caixa
            fv_ev = eq_value / dados.shares_outstanding
            if fv_ev > 0:
                valores.append(fv_ev)

        if not valores:
            return 0.0

        return float(np.mean(valores))

    # =========================================================================
    # Piotroski F-Score
    # =========================================================================

    def piotroski_f_score(
        self,
        dados: DadosFundamentalistas,
        dados_anterior: Optional[DadosFundamentalistas] = None,
    ) -> int:
        """
        Piotroski F-Score (0-9) — mede força fundamentalista.

        Critérios (1 ponto cada):
          LUCRATIVIDADE (0-4):
            1. ROA positivo
            2. FCO positivo
            3. ROA crescente (vs período anterior)
            4. FCO > Lucro Líquido (accruals)

          ALAVANCAGEM / LIQUIDEZ (0-3):
            5. Dívida/Ativos decrescente
            6. Current Ratio crescente
            7. Sem emissão de novas ações

          EFICIÊNCIA (0-2):
            8. Margem Bruta crescente
            9. Giro do ativo crescente (Receita/Ativos)

        Interpretação:
          0-3: FRACO (evitar)
          4-6: NEUTRO
          7-9: FORTE (comprar)
        """
        score = 0

        # 1. ROA positivo
        if dados.roa > 0:
            score += 1

        # 2. FCO positivo
        if dados.fluxo_caixa_operacional > 0:
            score += 1

        # 3. ROA crescente
        if dados_anterior and dados.roa > dados_anterior.roa:
            score += 1
        elif dados_anterior is None and dados.roa > 0:
            score += 1  # Sem dado anterior, dá ponto se positivo

        # 4. Accruals: FCO > Lucro (qualidade dos lucros)
        if dados.fluxo_caixa_operacional > dados.lucro_liquido:
            score += 1

        # 5. Dívida/Ativos decrescente
        alavancagem = dados.divida_bruta / dados.ativo_total if dados.ativo_total > 0 else 0
        if dados_anterior:
            alav_ant = (
                dados_anterior.divida_bruta / dados_anterior.ativo_total
                if dados_anterior.ativo_total > 0 else 0
            )
            if alavancagem < alav_ant:
                score += 1
        elif alavancagem < 0.5:
            score += 1

        # 6. Current Ratio crescente
        cr = (
            dados.ativo_circulante / dados.passivo_circulante
            if dados.passivo_circulante > 0 else 0
        )
        if dados_anterior:
            cr_ant = (
                dados_anterior.ativo_circulante / dados_anterior.passivo_circulante
                if dados_anterior.passivo_circulante > 0 else 0
            )
            if cr > cr_ant:
                score += 1
        elif cr > 1.0:
            score += 1

        # 7. Sem diluição (shares_outstanding não aumentou)
        if dados_anterior:
            if dados.shares_outstanding <= dados_anterior.shares_outstanding:
                score += 1
        else:
            score += 1  # Sem dado anterior, assume sem diluição

        # 8. Margem Bruta crescente
        if dados_anterior and dados.margem_bruta > dados_anterior.margem_bruta:
            score += 1
        elif dados_anterior is None and dados.margem_bruta > 0.2:
            score += 1

        # 9. Giro do ativo crescente (Receita/Ativo Total)
        giro = dados.receita_liquida / dados.ativo_total if dados.ativo_total > 0 else 0
        if dados_anterior:
            giro_ant = (
                dados_anterior.receita_liquida / dados_anterior.ativo_total
                if dados_anterior.ativo_total > 0 else 0
            )
            if giro > giro_ant:
                score += 1
        elif giro > 0.5:
            score += 1

        logger.info("piotroski_score", ticker=dados.ticker, score=score)
        return score

    # =========================================================================
    # Altman Z-Score
    # =========================================================================

    def altman_z_score(self, dados: DadosFundamentalistas) -> float:
        """
        Altman Z-Score — indicador de risco de falência.

        Z = 1.2×A + 1.4×B + 3.3×C + 0.6×D + 1.0×E

        Onde:
          A = Capital de Giro / Ativo Total
          B = Lucros Retidos / Ativo Total  (usa PL como proxy)
          C = EBIT / Ativo Total
          D = Market Cap / Passivo Total
          E = Receita / Ativo Total

        Interpretação:
          Z > 2.99: SEGURO (zona verde)
          1.81 < Z < 2.99: CINZA (zona de incerteza)
          Z < 1.81: PERIGO (risco de falência)

        NOTA: Modelo original para manufatura. Use com cautela para
              serviços e financeiras.
        """
        at = dados.ativo_total
        if at == 0:
            return 0.0

        a = dados.capital_giro / at
        b = dados.patrimonio_liquido / at  # proxy: Lucros Retidos ≈ PL
        c = dados.ebit / at
        d = dados.market_cap / dados.passivo_total if dados.passivo_total > 0 else 0
        e = dados.receita_liquida / at

        z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e

        zona = "SEGURO" if z > 2.99 else ("CINZA" if z > 1.81 else "PERIGO")
        logger.info(
            "altman_z_score",
            ticker=dados.ticker,
            z_score=round(z, 4),
            zona=zona,
        )

        return float(z)

    # =========================================================================
    # WACC e Custo de Capital
    # =========================================================================

    def calcular_wacc(
        self,
        dados: DadosFundamentalistas,
        rf: float = 0.1175,     # Selic (referência BR)
        erp: float = 0.06,     # Equity Risk Premium BR
        beta: float = 1.0,
        custo_divida: float = 0.12,
        aliquota_ir: float = 0.34,
    ) -> float:
        """
        WACC — Weighted Average Cost of Capital.

        WACC = ke × (E / (E+D)) + kd × (1-t) × (D / (E+D))

        Args:
            dados: Dados fundamentalistas (usa market_cap e dívida)
            rf: Taxa livre de risco
            erp: Prêmio de risco de mercado
            beta: Beta do ativo
            custo_divida: Custo da dívida bruto
            aliquota_ir: Alíquota efetiva de IR

        Returns:
            WACC em decimal
        """
        ke = rf + beta * erp

        equity = dados.market_cap
        debt = max(0, dados.divida_liquida)
        total = equity + debt

        if total == 0:
            return ke

        peso_e = equity / total
        peso_d = debt / total

        wacc = ke * peso_e + custo_divida * (1 - aliquota_ir) * peso_d

        logger.info(
            "wacc_calculado",
            ticker=dados.ticker,
            wacc=round(wacc, 4),
            ke=round(ke, 4),
            peso_equity=round(peso_e, 4),
        )

        return float(wacc)

    def custo_capital_capm(
        self,
        dados: DadosFundamentalistas,
        rf: float = 0.1175,
        erp: float = 0.06,
        beta: float = 1.0,
    ) -> float:
        """
        Custo do Capital Próprio via CAPM.

        ke = rf + β × (E[Rm] - rf)
        """
        return float(rf + beta * erp)

    # =========================================================================
    # Valuation Completo
    # =========================================================================

    def valuation_completo(
        self,
        dados: DadosFundamentalistas,
        dados_anterior: Optional[DadosFundamentalistas] = None,
        taxa_crescimento_fcf: float = 0.05,
        taxa_crescimento_div: float = 0.04,
        pe_justo: float = 15.0,
        ev_ebitda_justo: float = 10.0,
    ) -> ResultadoValuation:
        """
        Executa valuation completo com todos os métodos e gera recomendação.

        Retorna ResultadoValuation consolidado com fair value composto
        (média ponderada dos métodos disponíveis).
        """
        logger.info("valuation_completo_inicio", ticker=dados.ticker)

        # Múltiplos
        multiplos = self.calcular_multiplos(dados)

        # DCF
        dcf_fv = self.dcf_fcff(dados, taxa_crescimento_5y=taxa_crescimento_fcf)

        # DDM
        ddm_fv = self.ddm_gordon(dados, taxa_crescimento_dividendo=taxa_crescimento_div)

        # Graham
        graham = self.graham_number(dados)

        # Múltiplos fair value
        mult_fv = self.fair_value_por_multiplos(dados, pe_justo, ev_ebitda_justo)

        # Piotroski
        f_score = self.piotroski_f_score(dados, dados_anterior)

        # Altman
        z_score = self.altman_z_score(dados)

        # Fair value composto (média ponderada dos métodos válidos)
        metodos_fv: list[tuple[float, float]] = []  # (valor, peso)
        if dcf_fv > 0:
            metodos_fv.append((dcf_fv, 0.35))
        if ddm_fv > 0:
            metodos_fv.append((ddm_fv, 0.15))
        if graham > 0:
            metodos_fv.append((graham, 0.20))
        if mult_fv > 0:
            metodos_fv.append((mult_fv, 0.30))

        fair_composto: Optional[float] = None
        upside: Optional[float] = None
        recomendacao = ""
        confianca = 0.0

        if metodos_fv:
            total_peso = sum(p for _, p in metodos_fv)
            fair_composto = sum(v * p for v, p in metodos_fv) / total_peso

            if dados.preco_atual > 0:
                upside = (fair_composto - dados.preco_atual) / dados.preco_atual

                if upside > 0.20:
                    recomendacao = "SUBVALORIZADO"
                elif upside < -0.15:
                    recomendacao = "SOBREVALORIZADO"
                else:
                    recomendacao = "JUSTO"

            # Confiança baseada em quantos métodos concordam
            confianca = min(1.0, len(metodos_fv) / 4)

            # Bonus de confiança por Piotroski forte
            if f_score >= 7:
                confianca = min(1.0, confianca + 0.1)
            elif f_score <= 3:
                confianca = max(0.0, confianca - 0.1)

        resultado = ResultadoValuation(
            ticker=dados.ticker,
            data_referencia=dados.data_referencia,
            preco_atual=dados.preco_atual,
            dcf_fair_value=round(dcf_fv, 2) if dcf_fv > 0 else None,
            ddm_fair_value=round(ddm_fv, 2) if ddm_fv > 0 else None,
            graham_number=round(graham, 2) if graham > 0 else None,
            multiplos_fair_value=round(mult_fv, 2) if mult_fv > 0 else None,
            fair_value_composto=round(fair_composto, 2) if fair_composto else None,
            upside_downside=round(upside, 4) if upside is not None else None,
            piotroski_f_score=f_score,
            altman_z_score=round(z_score, 4),
            pe_ratio=multiplos.get("pe_ratio"),
            pb_ratio=multiplos.get("pb_ratio"),
            ev_ebitda=multiplos.get("ev_ebitda"),
            ev_fcf=multiplos.get("ev_fcf"),
            earnings_yield=multiplos.get("earnings_yield"),
            fcf_yield=multiplos.get("fcf_yield"),
            dividend_yield=multiplos.get("dividend_yield"),
            recomendacao=recomendacao,
            confianca=round(confianca, 2),
        )

        logger.info(
            "valuation_completo_fim",
            ticker=dados.ticker,
            fair_composto=fair_composto,
            upside=upside,
            recomendacao=recomendacao,
            piotroski=f_score,
            altman=round(z_score, 2),
        )

        return resultado

    # =========================================================================
    # Ranking Relativo (Comparação entre Ativos)
    # =========================================================================

    def ranking_relativo(
        self,
        lista_dados: list[DadosFundamentalistas],
    ) -> pd.DataFrame:
        """
        Ranking relativo entre múltiplos ativos por múltiplos de valuation.

        Ordena por valor composto (quanto menor múltiplos + mais qualidade, melhor).

        Retorna DataFrame com ranking e scores normalizados.
        """
        resultados: list[dict] = []

        for dados in lista_dados:
            multiplos = self.calcular_multiplos(dados)
            f_score = self.piotroski_f_score(dados)
            z_score = self.altman_z_score(dados)

            resultados.append({
                "ticker": dados.ticker,
                "preco": dados.preco_atual,
                "market_cap": dados.market_cap,
                **multiplos,
                "roe": round(dados.roe, 4),
                "roic": round(dados.roic, 4),
                "margem_liquida": round(dados.margem_liquida, 4),
                "margem_fcf": round(dados.margem_fcf, 4),
                "piotroski": f_score,
                "altman_z": round(z_score, 4),
            })

        df = pd.DataFrame(resultados)

        # Score composto: menor PE + menor EV/EBITDA + maior FCF Yield + maior Piotroski
        if not df.empty and "pe_ratio" in df.columns:
            # Normalizar rankings (rank ascendente = melhor para yields, descendente para múltiplos)
            for col in ["pe_ratio", "pb_ratio", "ev_ebitda"]:
                if col in df.columns and df[col].notna().any():
                    df[f"rank_{col}"] = df[col].rank(ascending=True, na_option="bottom")

            for col in ["fcf_yield", "earnings_yield", "roe", "piotroski"]:
                if col in df.columns and df[col].notna().any():
                    df[f"rank_{col}"] = df[col].rank(ascending=False, na_option="bottom")

            rank_cols = [c for c in df.columns if c.startswith("rank_")]
            if rank_cols:
                df["score_composto"] = df[rank_cols].mean(axis=1)
                df = df.sort_values("score_composto")

        logger.info("ranking_relativo", total_ativos=len(df))
        return df

    # =========================================================================
    # Utilitários
    # =========================================================================

    @staticmethod
    def _ultimo_valor(df: Optional[pd.DataFrame], label: str) -> float:
        """Extrai último valor disponível de um DataFrame de demonstrações."""
        if df is None or df.empty or label not in df.index:
            return 0.0
        valores = df.loc[label].dropna()
        if valores.empty:
            return 0.0
        return float(valores.iloc[0])
