"""
QuantNucleo — CLI Principal
============================
Interface de linha de comando (typer) para operações do núcleo quant.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="quantnucleo",
    help="🏛️ Núcleo Quantitativo LMF/UFU — CLI",
    add_completion=False,
)
console = Console()


@app.command()
def status() -> None:
    """Exibe status geral do núcleo quant."""
    from config.settings import get_config

    cfg = get_config()
    table = Table(title="📊 QuantNucleo — Status")
    table.add_column("Parâmetro", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Ambiente", cfg.ambiente.value)
    table.add_row("Versão", cfg.version)
    table.add_row("Debug", str(cfg.debug))
    table.add_row("Log Level", cfg.log_level)
    table.add_row("Azure RG", cfg.azure.resource_group)
    table.add_row("MongoDB DB", cfg.mongo.database)
    table.add_row("Risk VaR", f"{cfg.risk.var_confidence:.0%}")
    table.add_row("LSTM Look-back", str(cfg.lstm.look_back))

    console.print(table)


@app.command()
def ingest(
    tickers: str = typer.Argument(..., help="Tickers separados por vírgula (ex: PETR4.SA,VALE3.SA)"),
    start: str = typer.Option("2020-01-01", help="Data início (YYYY-MM-DD)"),
    end: str = typer.Option("", help="Data fim (YYYY-MM-DD). Vazio = hoje."),
) -> None:
    """Ingere dados de mercado para o data lake."""
    console.print(f"📡 Ingerindo dados: {tickers}")
    console.print(f"   Período: {start} → {end or 'hoje'}")
    # TODO: Implementar ingestão real via libs.data
    console.print("✅ Ingestão concluída")


@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="Nome da estratégia"),
    start: str = typer.Option("2020-01-01", help="Data início"),
    end: str = typer.Option("", help="Data fim"),
) -> None:
    """Executa backtest de uma estratégia."""
    console.print(f"🧪 Executando backtest: {strategy}")
    console.print(f"   Período: {start} → {end or 'hoje'}")
    # TODO: Implementar backtesting via services.backtest_runner
    console.print("✅ Backtest concluído")


@app.command()
def risk_report() -> None:
    """Gera relatório de risco do portfolio atual."""
    console.print("🛡️ Gerando relatório de risco...")
    # TODO: Implementar via libs.risk
    console.print("✅ Relatório gerado")


# =============================================================================
# Comandos de Renda Fixa
# =============================================================================


@app.command()
def rf_mtm(
    taxa_mercado: float = typer.Option(0.12, help="Taxa de mercado atual (decimal)"),
) -> None:
    """Calcula marcação a mercado para títulos de renda fixa de exemplo."""
    from datetime import date, timedelta

    from libs.fixed_income.titulos import RendaFixaEngine, TipoTitulo, TituloRendaFixa

    engine = RendaFixaEngine()
    hoje = date.today()

    titulos = [
        TituloRendaFixa(
            nome="LTN 2026",
            tipo=TipoTitulo.PREFIXADO,
            valor_nominal=1000.0,
            taxa_cupom=0.0,
            taxa_mercado=0.1175,
            vencimento=hoje + timedelta(days=504),
        ),
        TituloRendaFixa(
            nome="NTN-B 2030",
            tipo=TipoTitulo.IPCA_MAIS,
            valor_nominal=1000.0,
            taxa_cupom=0.06,
            taxa_mercado=0.055,
            vencimento=hoje + timedelta(days=1260),
        ),
    ]

    table = Table(title="📊 Marcação a Mercado — Renda Fixa")
    table.add_column("Título", style="cyan")
    table.add_column("PU Curva", style="green", justify="right")
    table.add_column("PU Mercado", style="yellow", justify="right")
    table.add_column("Δ %", justify="right")
    table.add_column("Duration Mod", justify="right")
    table.add_column("DV01", justify="right")

    for titulo in titulos:
        resultado = engine.marcacao_a_mercado(titulo, taxa_mercado=taxa_mercado)
        dur_mod = engine.duration_modificada(titulo)
        dv01 = engine.dv01(titulo)
        cor = "green" if resultado.diferenca_percentual >= 0 else "red"
        table.add_row(
            titulo.nome,
            f"R$ {resultado.pu_curva:,.2f}",
            f"R$ {resultado.pu_mercado:,.2f}",
            f"[{cor}]{resultado.diferenca_percentual:+.2%}[/{cor}]",
            f"{dur_mod:.4f}",
            f"R$ {dv01:,.4f}",
        )

    console.print(table)


@app.command()
def rf_cenarios(
    taxa_base: float = typer.Option(0.12, help="Taxa base do título"),
) -> None:
    """Análise de cenários de taxa de juros para renda fixa."""
    from datetime import date, timedelta

    from libs.fixed_income.titulos import RendaFixaEngine, TipoTitulo, TituloRendaFixa

    engine = RendaFixaEngine()

    titulo = TituloRendaFixa(
        nome="LTN Cenários",
        tipo=TipoTitulo.PREFIXADO,
        valor_nominal=1000.0,
        taxa_cupom=0.0,
        taxa_mercado=taxa_base,
        vencimento=date.today() + timedelta(days=504),
    )

    df = engine.cenarios_taxa(titulo, choques_bps=[-200, -100, -50, 0, 50, 100, 200])

    table = Table(title="📈 Cenários de Taxa — Renda Fixa")
    table.add_column("Choque (bps)", justify="right")
    table.add_column("Taxa", justify="right")
    table.add_column("PU", justify="right", style="green")
    table.add_column("Δ PU %", justify="right")

    pu_base = df[df["choque_bps"] == 0]["pu"].values[0] if 0 in df["choque_bps"].values else 0

    for _, row in df.iterrows():
        delta = ((row["pu"] / pu_base) - 1) if pu_base else 0
        cor = "green" if delta >= 0 else "red"
        table.add_row(
            f"{row['choque_bps']:+.0f}",
            f"{row['taxa']:.4%}",
            f"R$ {row['pu']:,.2f}",
            f"[{cor}]{delta:+.2%}[/{cor}]",
        )

    console.print(table)


# =============================================================================
# Comandos de Valuation
# =============================================================================


@app.command()
def valuation(
    ticker: str = typer.Argument(..., help="Ticker do ativo (ex: PETR4.SA, AAPL)"),
) -> None:
    """Executa valuation completo de uma ação."""
    from libs.valuation.engine import ValuationEngine

    engine = ValuationEngine()

    console.print(f"🔍 Coletando dados de [cyan]{ticker}[/cyan]...")
    try:
        dados = engine.coletar_dados_yfinance(ticker)
    except Exception as e:
        console.print(f"❌ Erro ao coletar dados: {e}")
        raise typer.Exit(code=1)

    resultado = engine.valuation_completo(dados)

    table = Table(title=f"💰 Valuation — {ticker}")
    table.add_column("Método / Métrica", style="cyan")
    table.add_column("Valor", style="green", justify="right")

    table.add_row("Preço Atual", f"R$ {resultado.preco_atual:,.2f}")
    table.add_row("", "")

    if resultado.dcf_fair_value:
        table.add_row("DCF (Fair Value)", f"R$ {resultado.dcf_fair_value:,.2f}")
    if resultado.ddm_fair_value:
        table.add_row("DDM Gordon", f"R$ {resultado.ddm_fair_value:,.2f}")
    if resultado.graham_number:
        table.add_row("Graham Number", f"R$ {resultado.graham_number:,.2f}")
    if resultado.multiplos_fair_value:
        table.add_row("Múltiplos (FV)", f"R$ {resultado.multiplos_fair_value:,.2f}")
    if resultado.fair_value_composto:
        table.add_row("[bold]Fair Value Composto[/bold]", f"[bold]R$ {resultado.fair_value_composto:,.2f}[/bold]")
    if resultado.upside_downside is not None:
        cor = "green" if resultado.upside_downside >= 0 else "red"
        table.add_row("Upside/Downside", f"[{cor}]{resultado.upside_downside:+.2%}[/{cor}]")

    table.add_row("", "")
    table.add_row("P/E", str(resultado.pe_ratio or "N/A"))
    table.add_row("EV/EBITDA", str(resultado.ev_ebitda or "N/A"))
    table.add_row("Earnings Yield", f"{resultado.earnings_yield:.2%}" if resultado.earnings_yield else "N/A")
    table.add_row("FCF Yield", f"{resultado.fcf_yield:.2%}" if resultado.fcf_yield else "N/A")
    table.add_row("Dividend Yield", f"{resultado.dividend_yield:.2%}" if resultado.dividend_yield else "N/A")

    table.add_row("", "")
    table.add_row("Piotroski F-Score", f"{resultado.piotroski_f_score}/9" if resultado.piotroski_f_score is not None else "N/A")
    table.add_row("Altman Z-Score", f"{resultado.altman_z_score:.2f}" if resultado.altman_z_score else "N/A")

    table.add_row("", "")
    cor_rec = {"SUBVALORIZADO": "green", "JUSTO": "yellow", "SOBREVALORIZADO": "red"}.get(resultado.recomendacao, "white")
    table.add_row("[bold]Recomendação[/bold]", f"[bold {cor_rec}]{resultado.recomendacao}[/bold {cor_rec}]")
    table.add_row("Confiança", f"{resultado.confianca:.0%}")

    console.print(table)


@app.command()
def ranking(
    tickers: str = typer.Argument(..., help="Tickers separados por vírgula (ex: PETR4.SA,VALE3.SA,BBAS3.SA)"),
) -> None:
    """Ranking comparativo de valuation entre múltiplos ativos."""
    from libs.valuation.engine import ValuationEngine

    engine = ValuationEngine()
    lista_tickers = [t.strip() for t in tickers.split(",")]

    console.print(f"📊 Coletando dados de {len(lista_tickers)} ativos...")
    dados_list = []
    for t in lista_tickers:
        try:
            dados = engine.coletar_dados_yfinance(t)
            dados_list.append(dados)
            console.print(f"  ✅ {t}")
        except Exception as e:
            console.print(f"  ❌ {t}: {e}")

    if not dados_list:
        console.print("❌ Nenhum ativo coletado com sucesso.")
        raise typer.Exit(code=1)

    df = engine.ranking_relativo(dados_list)

    table = Table(title="🏆 Ranking Relativo de Valuation")
    cols = ["ticker", "preco", "pe_ratio", "ev_ebitda", "fcf_yield", "roe", "piotroski", "score_composto"]
    for col in cols:
        if col in df.columns:
            table.add_column(col, justify="right" if col != "ticker" else "left")

    for _, row in df.iterrows():
        valores = []
        for col in cols:
            if col in df.columns:
                v = row[col]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    valores.append("N/A")
                elif isinstance(v, float):
                    valores.append(f"{v:.2f}")
                else:
                    valores.append(str(v))
        table.add_row(*valores)

    console.print(table)


if __name__ == "__main__":
    app()
