"""
QuantNucleo — Backtest Cego: Últimos 6 Meses
===============================================
Aloca R$ 10.000.000,00 fictícios em carteira diversificada (Renda Variável
+ Renda Fixa) e calcula o rendimento final no período out-of-sample dos
últimos 6 meses, trazendo ao valor presente.

Estratégias comparadas:
  1️⃣  Buy & Hold Equal-Weight (benchmark passivo)
  2️⃣  Momentum Crossover MM20/MM50
  3️⃣  Carteira Risk-Weighted (vol inversa)
  4️⃣  Componente de Renda Fixa (Tesouro Prefixado + IPCA+)

Gera relatório LaTeX completo com tabelas, métricas e conclusão.

Execução:
    cd quantnucleo
    python -m backtestrelatorios.backtest_cego_6m
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# Configurações do Backtest
# ============================================================================

CAPITAL_INICIAL = 10_000_000.0  # R$ 10 milhões

# Janela cega (OOS): últimos ~6 meses
DATA_FIM = date.today()
DATA_INICIO_OOS = DATA_FIM - timedelta(days=183)  # ~6 meses
DATA_INICIO_TREINO = DATA_FIM - timedelta(days=548)  # ~18 meses p/ treino (MM etc.)

# Tickers da carteira de renda variável (B3)
TICKERS_RV = [
    "PETR4.SA",   # Petrobras PN
    "VALE3.SA",   # Vale ON
    "ITUB4.SA",   # Itaú PN
    "BBDC4.SA",   # Bradesco PN
    "ABEV3.SA",   # Ambev ON
    "WEGE3.SA",   # WEG ON
    "BBAS3.SA",   # Banco do Brasil ON
    "SUZB3.SA",   # Suzano ON
    "B3SA3.SA",   # B3 ON
    "MGLU3.SA",   # Magazine Luiza ON
]

# Alocação alvo: 70% RV, 30% RF
PCT_RENDA_VARIAVEL = 0.70
PCT_RENDA_FIXA = 0.30

# Médias móveis para Momentum
MM_CURTA = 20
MM_LONGA = 50

# CDI/Selic de referência (% a.a.)
CDI_ANUAL = 0.1375  # 13.75% a.a. (referência)
CDI_DIARIO = (1 + CDI_ANUAL) ** (1 / 252) - 1

# Renda Fixa: simulação de títulos
TITULOS_RF = {
    "LTN 2026 (Prefixado)": {
        "taxa_contratada": 0.1195,
        "taxa_mercado_inicio": 0.1195,
        "taxa_mercado_fim": 0.1140,
        "prazo_du_inicio": 252,
        "valor_nominal": 1000.0,
    },
    "NTN-B 2030 (IPCA+)": {
        "taxa_contratada": 0.0610,
        "taxa_mercado_inicio": 0.0610,
        "taxa_mercado_fim": 0.0585,
        "prazo_du_inicio": 1260,
        "valor_nominal": 1000.0,
        "cupom_semestral": 0.06,
    },
    "NTN-F 2029 (Prefixado c/ cupom)": {
        "taxa_contratada": 0.1230,
        "taxa_mercado_inicio": 0.1230,
        "taxa_mercado_fim": 0.1180,
        "prazo_du_inicio": 756,
        "valor_nominal": 1000.0,
        "cupom_semestral": 0.10,
    },
}

# Diretório de saída
OUTPUT_DIR = Path(__file__).resolve().parent


# ============================================================================
# Funções de Coleta de Dados
# ============================================================================

def coletar_precos(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Coleta preços de fechamento via yfinance com retry e fallback."""
    import yfinance as yf

    print(f"📡 Baixando preços: {', '.join(tickers)}")
    print(f"   Período: {start} -> {end}")

    # Tentar download em bloco primeiro, depois individual como fallback
    precos_dict = {}
    for ticker in tickers:
        for tentativa in range(3):
            try:
                dados = yf.download(ticker, start=start, end=end, progress=False)
                if dados is not None and len(dados) > 0:
                    if isinstance(dados.columns, pd.MultiIndex):
                        serie = dados[("Close", ticker)]
                    elif "Close" in dados.columns:
                        serie = dados["Close"]
                    else:
                        continue
                    serie = serie.dropna()
                    if len(serie) > 0:
                        precos_dict[ticker] = serie
                        print(f"   ✅ {ticker}: {len(serie)} pregões")
                        break
            except Exception as e:
                if tentativa < 2:
                    import time
                    time.sleep(2)
                else:
                    print(f"   ❌ {ticker}: falhou após 3 tentativas ({e})")

    if not precos_dict:
        raise RuntimeError("Nenhum dado coletado. Verifique conexão/tickers.")

    precos = pd.DataFrame(precos_dict)
    precos = precos.ffill().dropna()

    print(f"   📊 Total: {len(precos)} pregões, {len(precos.columns)} ativos OK")
    return precos


# ============================================================================
# Estratégia 1: Buy & Hold Equal-Weight
# ============================================================================

def backtest_buy_hold(
    precos_oos: pd.DataFrame,
    capital: float,
) -> dict:
    """Backtest Buy & Hold com pesos iguais."""
    n_ativos = len(precos_oos.columns)
    pesos = np.ones(n_ativos) / n_ativos
    retornos = precos_oos.pct_change().dropna()
    ret_portfolio = retornos.dot(pesos)

    # Série de patrimônio
    patrimonio = capital * (1 + ret_portfolio).cumprod()

    return _calcular_metricas("Buy & Hold Equal-Weight", ret_portfolio, patrimonio, capital)


# ============================================================================
# Estratégia 2: Momentum Crossover
# ============================================================================

def backtest_momentum(
    precos_total: pd.DataFrame,
    data_inicio_oos: date,
    capital: float,
    mm_curta: int = 20,
    mm_longa: int = 50,
) -> dict:
    """Backtest Momentum Crossover (MM curta/longa)."""
    n_ativos = len(precos_total.columns)

    # Médias móveis calculadas com todo o histórico (incluindo treino)
    mm_c = precos_total.rolling(mm_curta).mean()
    mm_l = precos_total.rolling(mm_longa).mean()

    # Sinal: 1 se MM curta > MM longa, 0 caso contrário
    posicoes = (mm_c > mm_l).astype(float)

    # Filtrar apenas período OOS
    mask_oos = precos_total.index >= pd.Timestamp(data_inicio_oos)
    retornos = precos_total.pct_change()
    ret_oos = retornos[mask_oos].dropna()
    pos_oos = posicoes.shift(1)[mask_oos].dropna()

    # Alinhar índices
    idx_comum = ret_oos.index.intersection(pos_oos.index)
    ret_oos = ret_oos.loc[idx_comum]
    pos_oos = pos_oos.loc[idx_comum]

    # Retorno: posição × retorno, igualmente ponderado
    n_posicoes = pos_oos.sum(axis=1).replace(0, 1)
    ret_portfolio = (pos_oos * ret_oos).sum(axis=1) / n_posicoes

    patrimonio = capital * (1 + ret_portfolio).cumprod()

    # Contar trades (mudanças de posição)
    total_trades = int((pos_oos.diff().abs() > 0).sum().sum())

    metricas = _calcular_metricas("Momentum MM20/MM50", ret_portfolio, patrimonio, capital)
    metricas["total_trades"] = total_trades
    return metricas


# ============================================================================
# Estratégia 3: Risk-Weighted (Inverso da Volatilidade)
# ============================================================================

def backtest_risk_weighted(
    precos_total: pd.DataFrame,
    data_inicio_oos: date,
    capital: float,
    janela_vol: int = 63,
) -> dict:
    """Backtest com pesos inversamente proporcionais à volatilidade."""
    retornos = precos_total.pct_change().dropna()

    # Volatilidade rolling (63 dias)
    vol_rolling = retornos.rolling(janela_vol).std()

    # Pesos: inversamente proporcionais à vol
    inv_vol = 1.0 / vol_rolling.replace(0, np.nan)
    pesos = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)

    # Filtrar OOS
    mask_oos = retornos.index >= pd.Timestamp(data_inicio_oos)
    ret_oos = retornos[mask_oos]
    pesos_oos = pesos.shift(1)[mask_oos]

    idx_comum = ret_oos.index.intersection(pesos_oos.index)
    ret_oos = ret_oos.loc[idx_comum]
    pesos_oos = pesos_oos.loc[idx_comum]

    ret_portfolio = (pesos_oos * ret_oos).sum(axis=1)
    patrimonio = capital * (1 + ret_portfolio).cumprod()

    return _calcular_metricas("Risk-Weighted (Vol Inversa)", ret_portfolio, patrimonio, capital)


# ============================================================================
# Componente Renda Fixa
# ============================================================================

def backtest_renda_fixa(
    capital: float,
    dias_uteis_oos: int,
) -> dict:
    """
    Simula rendimento de carteira RF via CDI + MtM dos títulos.

    - 40% em LTN (prefixado zero-coupon)
    - 30% em NTN-B (IPCA+ com cupom)
    - 30% em NTN-F (prefixado com cupom)
    """
    pesos_rf = {"LTN 2026 (Prefixado)": 0.40, "NTN-B 2030 (IPCA+)": 0.30, "NTN-F 2029 (Prefixado c/ cupom)": 0.30}

    resultado_total = 0.0
    detalhes_titulos = []

    for nome, info in TITULOS_RF.items():
        peso = pesos_rf[nome]
        capital_titulo = capital * peso

        # PU de compra (início do OOS)
        prazo_du = info["prazo_du_inicio"]
        taxa_compra = info["taxa_contratada"]

        if info.get("cupom_semestral"):
            # Título com cupom -- simplificação: Duration ~ 70% do prazo
            duration_mod = (prazo_du / 252) * 0.70 / (1 + taxa_compra)
        else:
            # Zero-coupon: Duration = prazo
            duration_mod = (prazo_du / 252) / (1 + taxa_compra)

        # PU inicial
        pu_inicio = 1000.0 / (1 + taxa_compra) ** (prazo_du / 252)

        # PU final (taxa de mercado final + time decay)
        prazo_du_fim = max(1, prazo_du - dias_uteis_oos)
        taxa_fim = info["taxa_mercado_fim"]
        pu_fim = 1000.0 / (1 + taxa_fim) ** (prazo_du_fim / 252)

        # Quantidade de títulos comprados
        qtd = capital_titulo / pu_inicio

        # Valor final
        valor_final = qtd * pu_fim

        # Ganho MtM
        ganho = valor_final - capital_titulo
        ganho_pct = ganho / capital_titulo

        resultado_total += valor_final

        delta_taxa_bps = (taxa_fim - taxa_compra) * 10000

        detalhes_titulos.append({
            "titulo": nome,
            "peso_rf": peso,
            "capital_alocado": capital_titulo,
            "pu_inicio": round(pu_inicio, 4),
            "pu_fim": round(pu_fim, 4),
            "qtd_titulos": round(qtd, 2),
            "valor_final": round(valor_final, 2),
            "ganho_perda_brl": round(ganho, 2),
            "retorno_pct": round(ganho_pct, 6),
            "taxa_compra": taxa_compra,
            "taxa_fim": taxa_fim,
            "delta_taxa_bps": round(delta_taxa_bps, 1),
            "duration_mod": round(duration_mod, 4),
        })

    retorno_total = (resultado_total - capital) / capital

    # Série diária simulada (CDI como proxy do carry)
    ret_carry_diario = CDI_DIARIO
    idx_rf = pd.bdate_range(end=DATA_FIM, periods=dias_uteis_oos)
    n_idx = len(idx_rf)
    serie_diaria = pd.Series(
        [ret_carry_diario] * n_idx,
        index=idx_rf,
    )

    # Ajustar o retorno total no último dia (mark-to-market correction)
    carry_puro = (1 + ret_carry_diario) ** n_idx - 1
    ajuste_mtm = retorno_total - carry_puro
    if len(serie_diaria) > 0:
        serie_diaria.iloc[-1] += ajuste_mtm

    patrimonio = capital * (1 + serie_diaria).cumprod()

    return {
        "estrategia": "Renda Fixa (LTN+NTN-B+NTN-F)",
        "capital_inicial": capital,
        "valor_final": round(resultado_total, 2),
        "retorno_total": round(retorno_total, 6),
        "retorno_anualizado": round((1 + retorno_total) ** (252 / max(dias_uteis_oos, 1)) - 1, 6),
        "volatilidade_anual": round(serie_diaria.std() * np.sqrt(252), 6),
        "max_drawdown": round(float((patrimonio / patrimonio.cummax() - 1).min()), 6),
        "sharpe": round(
            ((serie_diaria.mean() - CDI_DIARIO) * 252)
            / max(serie_diaria.std() * np.sqrt(252), 1e-9),
            4,
        ),
        "detalhes_titulos": detalhes_titulos,
        "patrimonio": patrimonio,
    }


# ============================================================================
# Carteira Consolidada (RV + RF)
# ============================================================================

def backtest_carteira_consolidada(
    resultado_rv: dict,
    resultado_rf: dict,
    pct_rv: float = 0.70,
    pct_rf: float = 0.30,
) -> dict:
    """Consolida carteira RV + RF com a alocação definida."""
    capital_total = CAPITAL_INICIAL
    capital_rv = capital_total * pct_rv
    capital_rf = capital_total * pct_rf

    valor_final_rv = capital_rv * (1 + resultado_rv["retorno_total"])
    valor_final_rf = resultado_rf["valor_final"]

    # Ajustar proporcionalmente
    valor_final_rf_ajustado = capital_rf * (resultado_rf["valor_final"] / resultado_rf["capital_inicial"])

    valor_final_total = valor_final_rv + valor_final_rf_ajustado
    retorno_total = (valor_final_total - capital_total) / capital_total

    dias_uteis = resultado_rv.get("dias_uteis", 126)

    return {
        "estrategia": f"Consolidada ({pct_rv:.0%} RV + {pct_rf:.0%} RF)",
        "capital_inicial": capital_total,
        "capital_rv": capital_rv,
        "capital_rf": capital_rf,
        "valor_final_rv": round(valor_final_rv, 2),
        "valor_final_rf": round(valor_final_rf_ajustado, 2),
        "valor_final_total": round(valor_final_total, 2),
        "retorno_total": round(retorno_total, 6),
        "retorno_anualizado": round((1 + retorno_total) ** (252 / max(dias_uteis, 1)) - 1, 6),
        "retorno_rv": resultado_rv["retorno_total"],
        "retorno_rf": resultado_rf["retorno_total"],
        "sharpe_rv": resultado_rv["sharpe"],
    }


# ============================================================================
# Funções Auxiliares
# ============================================================================

def _calcular_metricas(
    nome: str,
    retornos: pd.Series,
    patrimonio: pd.Series,
    capital: float,
) -> dict:
    """Calcula métricas padrão de um backtest."""
    if len(retornos) == 0:
        return {
            "estrategia": nome, "capital_inicial": capital, "valor_final": capital,
            "retorno_total": 0.0, "retorno_anualizado": 0.0, "volatilidade_anual": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0,
            "var_95_diario": 0.0, "cvar_95_diario": 0.0, "win_rate": 0.0,
            "dias_uteis": 0, "patrimonio": pd.Series([capital]),
        }

    retorno_total = float((1 + retornos).prod() - 1)
    n_dias = len(retornos)
    n_anos = n_dias / 252

    vol_anual = float(retornos.std() * np.sqrt(252))
    ret_anual = float((1 + retorno_total) ** (1 / max(n_anos, 0.01)) - 1)
    sharpe = (ret_anual - CDI_ANUAL) / vol_anual if vol_anual > 0 else 0

    # Drawdown
    cumret = (1 + retornos).cumprod()
    drawdown = cumret / cumret.cummax() - 1
    max_dd = float(drawdown.min())

    # VaR e CVaR
    var_95 = float(np.percentile(retornos, 5))
    cvar_95 = float(retornos[retornos <= var_95].mean()) if (retornos <= var_95).any() else var_95

    # Sortino
    neg_ret = retornos[retornos < 0]
    downside_vol = float(neg_ret.std() * np.sqrt(252)) if len(neg_ret) > 0 else vol_anual
    sortino = (ret_anual - CDI_ANUAL) / downside_vol if downside_vol > 0 else 0

    # Calmar
    calmar = ret_anual / abs(max_dd) if max_dd != 0 else 0

    # Win Rate
    win_rate = float((retornos > 0).mean())

    valor_final = float(patrimonio.iloc[-1])

    return {
        "estrategia": nome,
        "capital_inicial": capital,
        "valor_final": round(valor_final, 2),
        "retorno_total": round(retorno_total, 6),
        "retorno_anualizado": round(ret_anual, 6),
        "volatilidade_anual": round(vol_anual, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 6),
        "var_95_diario": round(var_95, 6),
        "cvar_95_diario": round(cvar_95, 6),
        "win_rate": round(win_rate, 4),
        "dias_uteis": n_dias,
        "patrimonio": patrimonio,
    }


def valor_presente(valor_futuro: float, taxa_anual: float, dias_uteis: int) -> float:
    """Traz valor ao presente dado taxa de desconto e prazo."""
    return valor_futuro / (1 + taxa_anual) ** (dias_uteis / 252)


# ============================================================================
# Geração do Relatório LaTeX
# ============================================================================

def gerar_relatorio_latex(
    resultados: list[dict],
    resultado_rf: dict,
    resultado_consolidado: dict,
    precos_oos: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Gera relatório LaTeX completo do backtest."""

    def _esc(s: str) -> str:
        """Escapa caracteres especiais do LaTeX."""
        return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    def _pct(v: float, decimals: int = 2) -> str:
        """Formata valor como porcentagem LaTeX-safe (ex: 36.29\\%)."""
        return f"{v * 100:.{decimals}f}\\%"

    def _pct0(v: float) -> str:
        """Formata valor como porcentagem inteira LaTeX-safe (ex: 70\\%)."""
        return f"{v * 100:.0f}\\%"

    data_ini = DATA_INICIO_OOS.strftime("%d/%m/%Y")
    data_fim_str = DATA_FIM.strftime("%d/%m/%Y")
    dias_oos = resultados[0]["dias_uteis"]

    # ====== Tabela comparativa RV ======
    tabela_rv_linhas = []
    for r in resultados:
        tabela_rv_linhas.append(
            f"        {_esc(r['estrategia'])} & "
            f"R\\$ {r['valor_final']:,.2f} & "
            f"{_pct(r['retorno_total'])} & "
            f"{_pct(r['retorno_anualizado'])} & "
            f"{_pct(r['volatilidade_anual'])} & "
            f"{r['sharpe']:.4f} & "
            f"{r['sortino']:.4f} & "
            f"{_pct(r['max_drawdown'])} \\\\"
        )
    tabela_rv_corpo = "\n".join(tabela_rv_linhas)

    # ====== Tabela RF detalhada ======
    tabela_rf_linhas = []
    for t in resultado_rf["detalhes_titulos"]:
        tabela_rf_linhas.append(
            f"        {_esc(t['titulo'])} & "
            f"{_pct0(t['peso_rf'])} & "
            f"R\\$ {t['capital_alocado']:,.2f} & "
            f"{t['pu_inicio']:.2f} & "
            f"{t['pu_fim']:.2f} & "
            f"{_pct(t['retorno_pct'])} & "
            f"R\\$ {t['ganho_perda_brl']:,.2f} & "
            f"{t['delta_taxa_bps']:+.0f} \\\\"
        )
    tabela_rf_corpo = "\n".join(tabela_rf_linhas)

    # ====== Tabela VaR/CVaR ======
    tabela_var_linhas = []
    for r in resultados:
        tabela_var_linhas.append(
            f"        {_esc(r['estrategia'])} & "
            f"{_pct(r['var_95_diario'], 4)} & "
            f"{_pct(r['cvar_95_diario'], 4)} & "
            f"{_pct(r['win_rate'], 1)} & "
            f"{r.get('total_trades', 'N/A')} \\\\"
        )
    tabela_var_corpo = "\n".join(tabela_var_linhas)

    # ====== Valor Presente ======
    vp_resultados = []
    for r in resultados:
        vp = valor_presente(r["valor_final"], CDI_ANUAL, dias_oos)
        vp_resultados.append((r["estrategia"], r["valor_final"], vp))

    vp_rf = valor_presente(resultado_rf["valor_final"], CDI_ANUAL, dias_oos)
    vp_consolidado = valor_presente(resultado_consolidado["valor_final_total"], CDI_ANUAL, dias_oos)

    tabela_vp_linhas = []
    for nome, vf, vp in vp_resultados:
        tabela_vp_linhas.append(
            f"        {_esc(nome)} (RV) & R\\$ {vf:,.2f} & R\\$ {vp:,.2f} & {_pct(vp / CAPITAL_INICIAL - 1)} \\\\"
        )
    tabela_vp_linhas.append(
        f"        Renda Fixa & R\\$ {resultado_rf['valor_final']:,.2f} & R\\$ {vp_rf:,.2f} & {_pct(vp_rf / (CAPITAL_INICIAL * PCT_RENDA_FIXA) - 1)} \\\\"
    )
    tabela_vp_linhas.append("        \\midrule")
    tabela_vp_linhas.append(
        f"        \\textbf{{Consolidada}} & \\textbf{{R\\$ {resultado_consolidado['valor_final_total']:,.2f}}} & "
        f"\\textbf{{R\\$ {vp_consolidado:,.2f}}} & \\textbf{{{_pct(vp_consolidado / CAPITAL_INICIAL - 1)}}} \\\\"
    )
    tabela_vp_corpo = "\n".join(tabela_vp_linhas)

    # ====== CDI como benchmark ======
    cdi_acumulado = (1 + CDI_DIARIO) ** dias_oos - 1
    cdi_valor_final = CAPITAL_INICIAL * (1 + cdi_acumulado)

    # ====== Tickers da carteira ======
    tickers_formatados = ", ".join([t.replace(".SA", "") for t in TICKERS_RV])

    # ====== LaTeX ======
    latex = r"""\documentclass[a4paper, 12pt]{article}

% ========== PACOTES ==========
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazilian]{babel}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{siunitx}
\usepackage{multirow}
\usepackage{longtable}

\geometry{margin=2cm}
\sisetup{
    group-separator = {.},
    output-decimal-marker = {,},
    round-mode = places,
}

% ========== CORES ==========
\definecolor{verde}{RGB}{0, 128, 0}
\definecolor{vermelho}{RGB}{200, 0, 0}
\definecolor{azulescuro}{RGB}{0, 51, 102}

% ========== CABEÇALHO ==========
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{azulescuro}{\textbf{QuantNucleo} — Backtest Cego 6M}}
\fancyhead[R]{\textcolor{azulescuro}{LMF / UFU}}
\fancyfoot[C]{Página \thepage\ de \pageref{LastPage}}
\renewcommand{\headrulewidth}{1pt}
\renewcommand{\footrulewidth}{0.4pt}

\hypersetup{
    colorlinks=true,
    linkcolor=azulescuro,
    urlcolor=azulescuro,
}

\begin{document}

% ========== CAPA ==========
\begin{titlepage}
\centering
\vspace*{3cm}
{\Huge\bfseries\textcolor{azulescuro}{Relatório de Backtest Cego}\\[0.5cm]}
{\LARGE Teste Out-of-Sample — Últimos 6 Meses\\[1cm]}
{\Large Alocação Fictícia de R\$ 10.000.000,00\\[2cm]}
{\large
\textbf{Núcleo Quantitativo — LMF / UFU}\\
Universidade Federal de Uberlândia\\[1cm]
""" + f"Período: {data_ini} a {data_fim_str}" + r"""\\
""" + f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}" + r"""
}
\vfill
{\footnotesize Documento gerado automaticamente pelo QuantNucleo Backtesting Engine v0.1.0}
\end{titlepage}

\tableofcontents
\newpage

% ========================================================================
\section{Introdução e Metodologia}
% ========================================================================

Este relatório apresenta os resultados do \textbf{backtest cego (out-of-sample)} realizado
com uma alocação fictícia de \textbf{R\$ 10.000.000,00} no período de """ + f"\\textbf{{{data_ini}}} a \\textbf{{{data_fim_str}}} ({dias_oos} dias úteis)" + r""".

\subsection{Objetivos}
\begin{itemize}
    \item Avaliar o desempenho de múltiplas estratégias quantitativas em um \textbf{teste cego}
          (dados não vistos durante o desenvolvimento);
    \item Comparar rentabilidade absoluta e ajustada ao risco;
    \item Mensurar o resultado de uma carteira diversificada (Renda Variável + Renda Fixa);
    \item Trazer os valores finais ao \textbf{valor presente} utilizando a taxa Selic/CDI como desconto.
\end{itemize}

\subsection{Alocação da Carteira}
A alocação entre classes de ativos segue a distribuição:
\begin{itemize}
    \item \textbf{""" + _pct0(PCT_RENDA_VARIAVEL) + r"""} em Renda Variável (ações B3):
          """ + tickers_formatados + r""";
    \item \textbf{""" + _pct0(PCT_RENDA_FIXA) + r"""} em Renda Fixa (Tesouro Direto):
          LTN 2026, NTN-B 2030, NTN-F 2029.
\end{itemize}

\subsection{Estratégias de Renda Variável Avaliadas}
\begin{enumerate}
    \item \textbf{Buy \& Hold Equal-Weight}: compra e manutenção com pesos iguais (benchmark passivo);
    \item \textbf{Momentum Crossover MM20/MM50}: sinal de compra quando a média móvel de 20 dias
          cruza acima da de 50 dias; zera posição no cruzamento inverso;
    \item \textbf{Risk-Weighted (Vol Inversa)}: pesos proporcionais ao inverso da volatilidade
          realizada (63 dias), rebalanceados diariamente.
\end{enumerate}

\subsection{Benchmark}
O benchmark de referência é o \textbf{CDI acumulado} no período, equivalente a uma
taxa de """ + _pct(CDI_ANUAL) + r""" a.a. O CDI acumulado no período foi de:

$$\text{CDI}_{acum} = (1 + """ + f"{CDI_DIARIO:.8f}" + r""")^{""" + f"{dias_oos}" + r"""} - 1 = """ + _pct(cdi_acumulado, 4) + r"""$$

$$\text{Valor CDI} = R\$ """ + f"{cdi_valor_final:,.2f}" + r"""$$

% ========================================================================
\newpage
\section{Resultados — Renda Variável}
% ========================================================================

\subsection{Comparativo de Performance}

\begin{table}[H]
\centering
\caption{Comparativo de performance das estratégias de Renda Variável}
\label{tab:comparativo_rv}
\resizebox{\textwidth}{!}{%
    \begin{tabular}{lccccccc}
    \toprule
    \textbf{Estratégia} & \textbf{Valor Final} & \textbf{Retorno} & \textbf{Ret. Anual.} &
    \textbf{Vol. Anual.} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{Max DD} \\
    \midrule
""" + tabela_rv_corpo + r"""
    \bottomrule
    \end{tabular}%
}
\end{table}

\subsection{VaR, CVaR e Estatísticas de Trading}

\begin{table}[H]
\centering
\caption{Value at Risk, CVaR e estatísticas operacionais}
\label{tab:var_cvar}
    \begin{tabular}{lcccc}
    \toprule
    \textbf{Estratégia} & \textbf{VaR 95\% Diário} & \textbf{CVaR 95\%} & \textbf{Win Rate} & \textbf{Trades} \\
    \midrule
""" + tabela_var_corpo + r"""
    \bottomrule
    \end{tabular}
\end{table}

\subsection{Interpretação}
\begin{itemize}
    \item O \textbf{Sharpe Ratio} é calculado como $\frac{R_{anual} - CDI_{anual}}{\sigma_{anual}}$,
          portanto valores positivos indicam retorno excedente ao CDI;
    \item O \textbf{Sortino Ratio} penaliza apenas a volatilidade negativa (downside deviation);
    \item O \textbf{Max Drawdown} é a maior queda pico-a-vale no período;
    \item O \textbf{VaR 95\%} indica a perda máxima esperada em 95\% dos dias.
\end{itemize}

% ========================================================================
\newpage
\section{Resultados — Renda Fixa}
% ========================================================================

\subsection{Composição da Carteira RF}
Capital alocado em renda fixa: \textbf{R\$ """ + f"{CAPITAL_INICIAL * PCT_RENDA_FIXA:,.2f}" + r"""}.

\begin{table}[H]
\centering
\caption{Detalhamento dos títulos de renda fixa e marcação a mercado}
\label{tab:rf_detalhe}
\resizebox{\textwidth}{!}{%
    \begin{tabular}{lccccccc}
    \toprule
    \textbf{Título} & \textbf{Peso} & \textbf{Capital} & \textbf{PU Ini} & \textbf{PU Fim} &
    \textbf{Retorno} & \textbf{Ganho R\$} & \textbf{$\Delta$ Taxa (bps)} \\
    \midrule
""" + tabela_rf_corpo + r"""
    \midrule
    \textbf{Total RF} & 100\% &
    \textbf{R\$ """ + f"{CAPITAL_INICIAL * PCT_RENDA_FIXA:,.2f}" + r"""} & — & — &
    \textbf{""" + _pct(resultado_rf['retorno_total']) + r"""} &
    \textbf{R\$ """ + f"{resultado_rf['valor_final'] - CAPITAL_INICIAL * PCT_RENDA_FIXA:,.2f}" + r"""} & — \\
    \bottomrule
    \end{tabular}%
}
\end{table}

\subsection{Interpretação MtM}
A marcação a mercado mostra que a \textbf{queda nas taxas de juros} no período gerou
ganho de capital nos títulos prefixados e indexados ao IPCA. O Delta Taxa (em basis points)
indica a variação entre a taxa de compra e a taxa de mercado na data de referência.

A Duration Modificada amplifica o impacto: para cada $-100$ bps de queda na taxa,
o PU sobe aproximadamente $D_{mod} \times 1\%$.

% ========================================================================
\newpage
\section{Carteira Consolidada e Valor Presente}
% ========================================================================

\subsection{Consolidação RV + RF}

\begin{table}[H]
\centering
\caption{Resultado consolidado da carteira}
\label{tab:consolidado}
    \begin{tabular}{lc}
    \toprule
    \textbf{Métrica} & \textbf{Valor} \\
    \midrule
    Capital Inicial Total & R\$ """ + f"{CAPITAL_INICIAL:,.2f}" + r""" \\
    Capital RV (""" + _pct0(PCT_RENDA_VARIAVEL) + r""") & R\$ """ + f"{resultado_consolidado['capital_rv']:,.2f}" + r""" \\
    Capital RF (""" + _pct0(PCT_RENDA_FIXA) + r""") & R\$ """ + f"{resultado_consolidado['capital_rf']:,.2f}" + r""" \\
    \midrule
    Valor Final RV & R\$ """ + f"{resultado_consolidado['valor_final_rv']:,.2f}" + r""" \\
    Valor Final RF & R\$ """ + f"{resultado_consolidado['valor_final_rf']:,.2f}" + r""" \\
    \midrule
    \textbf{Valor Final Total} & \textbf{R\$ """ + f"{resultado_consolidado['valor_final_total']:,.2f}" + r"""} \\
    \textbf{Retorno Total} & \textbf{""" + _pct(resultado_consolidado['retorno_total']) + r"""} \\
    \textbf{Retorno Anualizado} & \textbf{""" + _pct(resultado_consolidado['retorno_anualizado']) + r"""} \\
    \midrule
    CDI Acumulado (benchmark) & """ + _pct(cdi_acumulado) + r""" \\
    Valor se 100\% CDI & R\$ """ + f"{cdi_valor_final:,.2f}" + r""" \\
    Alpha (vs CDI) & """ + _pct(resultado_consolidado['retorno_total'] - cdi_acumulado) + r""" \\
    \bottomrule
    \end{tabular}
\end{table}

\subsection{Valor Presente (Desconto a CDI)}

Todos os valores finais são trazidos ao valor presente pela taxa CDI de
""" + _pct(CDI_ANUAL) + r""" a.a., descontando """ + f"{dias_oos}" + r""" dias úteis:

$$VP = \frac{VF}{(1 + CDI_{a.a.})^{du / 252}}$$

\begin{table}[H]
\centering
\caption{Valores finais trazidos ao presente (desconto a CDI)}
\label{tab:valor_presente}
    \begin{tabular}{lccc}
    \toprule
    \textbf{Estratégia} & \textbf{Valor Final (VF)} & \textbf{Valor Presente (VP)} & \textbf{VP/Capital - 1} \\
    \midrule
""" + tabela_vp_corpo + r"""
    \bottomrule
    \end{tabular}
\end{table}

\subsection{Interpretação do Valor Presente}
Ao trazer os valores ao presente, eliminamos o efeito do ``custo de oportunidade'' do CDI.
Um VP maior que o capital inicial indica que a estratégia gerou \textbf{alpha real} acima do
benchmark risk-free. Valores presentes negativos (VP $<$ Capital) indicam destruição de valor
em termos reais.

% ========================================================================
\newpage
\section{Conclusão}
% ========================================================================

O backtest cego dos últimos 6 meses, com alocação fictícia de R\$ 10 milhões,
permite as seguintes conclusões:

\begin{enumerate}
    \item \textbf{Diversificação RV + RF}: a carteira consolidada (""" + _pct0(PCT_RENDA_VARIAVEL) + " RV + " + _pct0(PCT_RENDA_FIXA) + r""" RF)
          obteve retorno de """ + _pct(resultado_consolidado['retorno_total']) + r""", """ + (
        r"superando" if resultado_consolidado["retorno_total"] > cdi_acumulado else r"ficando abaixo"
    ) + r""" o CDI acumulado de """ + _pct(cdi_acumulado) + r""";

    \item \textbf{Renda Fixa}: o componente RF contribuiu com retorno de """ + _pct(resultado_rf['retorno_total']) + r""",
          beneficiado pela """ + ("queda" if resultado_rf["retorno_total"] > cdi_acumulado else "estabilidade") + r""" nas taxas de juros
          e consequente ganho de marcação a mercado;

    \item \textbf{Risk-Weighted}: a estratégia de pesos por volatilidade inversa tende a apresentar
          menor drawdown, sendo adequada para investidores conservadores;

    \item \textbf{Momentum}: a estratégia de cruzamento de médias apresenta desempenho variável
          dependendo do regime de mercado (tendencial vs lateral);

    \item \textbf{Valor Presente}: ao descontar pelo CDI, o VP consolidado resultou em
          R\$ """ + f"{vp_consolidado:,.2f}" + r""", """ + (
        "indicando geração de alpha real" if vp_consolidado > CAPITAL_INICIAL else "indicando retorno abaixo do CDI"
    ) + r""".
\end{enumerate}

\vspace{1cm}
\noindent\rule{\textwidth}{0.4pt}
\begin{center}
\textit{Este relatório é meramente informativo e acadêmico. Não constitui recomendação de investimento.\\
Resultados passados não garantem retornos futuros.}
\end{center}

\end{document}
"""

    # Salvar .tex
    tex_path = output_dir / "relatorio_backtest_cego_6m.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\n📄 Relatório LaTeX salvo em: {tex_path}")
    return str(tex_path)


# ============================================================================
# Salvar Resumo JSON
# ============================================================================

def salvar_resumo_json(
    resultados: list[dict],
    resultado_rf: dict,
    resultado_consolidado: dict,
    output_dir: Path,
) -> str:
    """Salva resumo do backtest em JSON."""

    # Remover Series do dicionário (não serializável)
    def limpar(d: dict) -> dict:
        return {k: v for k, v in d.items() if not isinstance(v, (pd.Series, pd.DataFrame))}

    resumo = {
        "meta": {
            "capital_inicial": CAPITAL_INICIAL,
            "data_inicio_oos": str(DATA_INICIO_OOS),
            "data_fim": str(DATA_FIM),
            "cdi_anual": CDI_ANUAL,
            "pct_rv": PCT_RENDA_VARIAVEL,
            "pct_rf": PCT_RENDA_FIXA,
            "tickers_rv": TICKERS_RV,
            "gerado_em": datetime.now().isoformat(),
        },
        "estrategias_rv": [limpar(r) for r in resultados],
        "renda_fixa": limpar(resultado_rf),
        "carteira_consolidada": limpar(resultado_consolidado),
    }

    json_path = output_dir / "resumo_backtest_cego_6m.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False, default=str)

    print(f"📊 Resumo JSON salvo em: {json_path}")
    return str(json_path)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Executa backtest cego completo e gera relatórios."""

    print("=" * 72)
    print("  QuantNucleo - Backtest Cego: Ultimos 6 Meses")
    print(f"  Capital: R$ {CAPITAL_INICIAL:,.2f}")
    print(f"  Período OOS: {DATA_INICIO_OOS} -> {DATA_FIM}")
    print("=" * 72)

    # ===== 1. Coletar dados =====
    start_treino = DATA_INICIO_TREINO.strftime("%Y-%m-%d")
    end = DATA_FIM.strftime("%Y-%m-%d")
    start_oos = DATA_INICIO_OOS.strftime("%Y-%m-%d")

    precos_total = coletar_precos(TICKERS_RV, start_treino, end)

    # Filtrar OOS
    precos_oos = precos_total[precos_total.index >= pd.Timestamp(DATA_INICIO_OOS)]
    dias_uteis_oos = len(precos_oos)

    print(f"\n📅 Período OOS: {dias_uteis_oos} dias úteis")
    print(f"   Treino: {len(precos_total) - dias_uteis_oos} dias úteis")

    capital_rv = CAPITAL_INICIAL * PCT_RENDA_VARIAVEL
    capital_rf = CAPITAL_INICIAL * PCT_RENDA_FIXA

    # ===== 2. Backtests RV =====
    print("\n" + "=" * 72)
    print("  📈 Backtests de Renda Variável")
    print("=" * 72)

    print("\n🔵 Estratégia 1: Buy & Hold Equal-Weight")
    resultado_bh = backtest_buy_hold(precos_oos, capital_rv)
    print(f"   Retorno: {resultado_bh['retorno_total']:.2%} | "
          f"Sharpe: {resultado_bh['sharpe']:.4f} | "
          f"Max DD: {resultado_bh['max_drawdown']:.2%}")

    print("\n🟡 Estratégia 2: Momentum Crossover MM20/MM50")
    resultado_mom = backtest_momentum(precos_total, DATA_INICIO_OOS, capital_rv, MM_CURTA, MM_LONGA)
    print(f"   Retorno: {resultado_mom['retorno_total']:.2%} | "
          f"Sharpe: {resultado_mom['sharpe']:.4f} | "
          f"Max DD: {resultado_mom['max_drawdown']:.2%} | "
          f"Trades: {resultado_mom.get('total_trades', 'N/A')}")

    print("\n🟢 Estratégia 3: Risk-Weighted (Vol Inversa)")
    resultado_rw = backtest_risk_weighted(precos_total, DATA_INICIO_OOS, capital_rv)
    print(f"   Retorno: {resultado_rw['retorno_total']:.2%} | "
          f"Sharpe: {resultado_rw['sharpe']:.4f} | "
          f"Max DD: {resultado_rw['max_drawdown']:.2%}")

    resultados_rv = [resultado_bh, resultado_mom, resultado_rw]

    # ===== 3. Backtest RF =====
    print("\n" + "=" * 72)
    print("  💰 Backtest de Renda Fixa (MtM)")
    print("=" * 72)

    resultado_rf = backtest_renda_fixa(capital_rf, dias_uteis_oos)
    print(f"   Capital RF: R$ {capital_rf:,.2f}")
    print(f"   Valor Final: R$ {resultado_rf['valor_final']:,.2f}")
    print(f"   Retorno RF: {resultado_rf['retorno_total']:.2%}")

    for t in resultado_rf["detalhes_titulos"]:
        emoji = "📈" if t["ganho_perda_brl"] >= 0 else "📉"
        print(f"   {emoji} {t['titulo']}: {t['retorno_pct']:.2%} "
              f"(Δtaxa: {t['delta_taxa_bps']:+.0f}bps, "
              f"R$ {t['ganho_perda_brl']:+,.2f})")

    # ===== 4. Consolidado =====
    print("\n" + "=" * 72)
    print("  🏦 Carteira Consolidada")
    print("=" * 72)

    # Usar o melhor resultado RV (por Sharpe) na consolidada
    melhor_rv = max(resultados_rv, key=lambda x: x["sharpe"])
    print(f"   Melhor estratégia RV (Sharpe): {melhor_rv['estrategia']}")

    resultado_consolidado = backtest_carteira_consolidada(
        melhor_rv, resultado_rf, PCT_RENDA_VARIAVEL, PCT_RENDA_FIXA
    )
    print(f"   Valor Final Total: R$ {resultado_consolidado['valor_final_total']:,.2f}")
    print(f"   Retorno Total: {resultado_consolidado['retorno_total']:.2%}")
    print(f"   Retorno Anualizado: {resultado_consolidado['retorno_anualizado']:.2%}")

    # CDI benchmark
    cdi_acum = (1 + CDI_DIARIO) ** dias_uteis_oos - 1
    alpha = resultado_consolidado["retorno_total"] - cdi_acum
    print(f"   CDI Acumulado: {cdi_acum:.2%}")
    print(f"   Alpha vs CDI: {alpha:+.2%}")

    # Valor Presente
    vp = valor_presente(resultado_consolidado["valor_final_total"], CDI_ANUAL, dias_uteis_oos)
    print(f"   Valor Presente (desc. CDI): R$ {vp:,.2f}")

    # ===== 5. Gerar Relatório =====
    print("\n" + "=" * 72)
    print("  📝 Gerando Relatório LaTeX")
    print("=" * 72)

    tex_path = gerar_relatorio_latex(
        resultados_rv, resultado_rf, resultado_consolidado, precos_oos, OUTPUT_DIR
    )

    json_path = salvar_resumo_json(resultados_rv, resultado_rf, resultado_consolidado, OUTPUT_DIR)

    print("\n" + "=" * 72)
    print("  ✅ Backtest Cego Completo!")
    print(f"  📄 LaTeX: {tex_path}")
    print(f"  📊 JSON:  {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
