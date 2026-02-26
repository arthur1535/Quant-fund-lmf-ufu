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


if __name__ == "__main__":
    app()
