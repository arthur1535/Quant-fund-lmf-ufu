"""
QuantNucleo — Ingestão de Dados de Mercado
============================================
Ingestão robusta com retry, fallback e versionamento de datasets.

Fontes:
 - yfinance (ações BR/US, futuros, ETFs)
 - BCB API (Selic, IPCA, CDI, câmbio)
 - CVM API (fundos brasileiros)

Destinos:
 - Local: data/raw/, data/processed/
 - Azure: Blob Storage (raw → processed → features)

Padrões:
 - Cache local para fallback se API falhar
 - Versionamento por timestamp no nome do arquivo
 - Retry com backoff exponencial
 - Logs estruturados para auditoria
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from libs.core.logging import get_logger

logger = get_logger("data.ingestion")

# Diretórios locais (dev)
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"


class MarketDataIngester:
    """
    Ingere dados de mercado de múltiplas fontes com retry e fallback.
    
    Uso:
        ingester = MarketDataIngester()
        df = ingester.fetch_prices(["PETR4.SA", "VALE3.SA"], "2020-01-01")
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_dir = cache_dir or RAW_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # yfinance — Ações, ETFs, Futuros
    # =========================================================================

    def fetch_prices(
        self,
        tickers: Sequence[str],
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Baixa preços via yfinance com retry e cache de fallback.
        
        Args:
            tickers: Lista de tickers (ex: ["PETR4.SA", "VALE3.SA"])
            start: Data início "YYYY-MM-DD"
            end: Data fim "YYYY-MM-DD" (None = hoje)
            interval: Intervalo (1d, 1wk, 1mo)
            
        Returns:
            DataFrame com preços de fechamento (Close)
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "fetch_prices_inicio",
            tickers=list(tickers),
            start=start,
            end=end,
            interval=interval,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                data = yf.download(
                    list(tickers),
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )

                if data is None or data.empty:
                    raise ValueError("yfinance retornou dados vazios")

                # Extrair Close
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Close"]
                else:
                    prices = data[["Close"]].rename(columns={"Close": tickers[0]})

                prices = prices.dropna(how="all")

                # Salvar cache
                self._save_cache(prices, "prices", tickers)

                logger.info(
                    "fetch_prices_sucesso",
                    linhas=len(prices),
                    colunas=len(prices.columns),
                    attempt=attempt,
                )
                return prices

            except Exception as e:
                logger.warning(
                    "fetch_prices_erro",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    erro=str(e),
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        # Fallback: cache local
        logger.warning("fetch_prices_fallback_cache")
        return self._load_cache("prices", tickers)

    # =========================================================================
    # BCB API — Dados Macroeconômicos Brasileiros
    # =========================================================================

    def fetch_bcb_series(
        self,
        codigo: int,
        data_inicial: str,
        data_final: Optional[str] = None,
        nome: str = "valor",
    ) -> pd.DataFrame:
        """
        Busca série temporal do Banco Central do Brasil.
        
        Códigos comuns:
            432  → Selic (meta)
            433  → IPCA
            1     → Dólar (venda)
            12    → CDI
            
        Args:
            codigo: Código da série BCB
            data_inicial: "DD/MM/YYYY"
            data_final: "DD/MM/YYYY" (None = hoje)
        """
        if data_final is None:
            data_final = datetime.now().strftime("%d/%m/%Y")

        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}"
            f"/dados?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
        )

        logger.info("fetch_bcb_inicio", codigo=codigo, url=url)

        for attempt in range(1, self.max_retries + 1):
            try:
                df = pd.read_json(url)
                df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
                df = df.set_index("data").rename(columns={"valor": nome})
                
                logger.info("fetch_bcb_sucesso", codigo=codigo, linhas=len(df))
                return df

            except Exception as e:
                logger.warning("fetch_bcb_erro", codigo=codigo, attempt=attempt, erro=str(e))
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"❌ Falha ao buscar série BCB {codigo} após {self.max_retries} tentativas")

    # =========================================================================
    # Transformações Financeiras
    # =========================================================================

    @staticmethod
    def calcular_retornos(
        precos: pd.DataFrame,
        tipo: str = "log",
    ) -> pd.DataFrame:
        """Calcula retornos diários (log ou simples)."""
        precos = precos.ffill().dropna()
        if tipo == "log":
            return np.log(precos / precos.shift(1)).dropna()
        else:
            return precos.pct_change().dropna()

    @staticmethod
    def calcular_cdi_diario(taxa_anual: float) -> float:
        """Converte CDI anual → diário (252 du/ano)."""
        return (1 + taxa_anual) ** (1 / 252) - 1

    @staticmethod
    def calcular_retorno_acumulado(retornos: pd.DataFrame) -> pd.DataFrame:
        """Retorno acumulado a partir de retornos simples."""
        return (1 + retornos).cumprod() - 1

    # =========================================================================
    # Cache / Versionamento
    # =========================================================================

    def _save_cache(
        self, df: pd.DataFrame, tipo: str, tickers: Sequence[str]
    ) -> None:
        """Salva cache local versionado por timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_str = "_".join(t.replace(".", "_") for t in tickers[:5])
        filename = f"{tipo}_{ticker_str}_{ts}.parquet"
        path = self.cache_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.debug("cache_salvo", path=str(path))

    def _load_cache(
        self, tipo: str, tickers: Sequence[str]
    ) -> pd.DataFrame:
        """Carrega cache mais recente disponível."""
        ticker_str = "_".join(t.replace(".", "_") for t in tickers[:5])
        pattern = f"{tipo}_{ticker_str}_*.parquet"
        files = sorted(self.cache_dir.glob(pattern), reverse=True)

        if not files:
            raise FileNotFoundError(
                f"❌ Nenhum cache encontrado para {tipo}/{ticker_str}"
            )

        path = files[0]
        logger.info("cache_carregado", path=str(path))
        return pd.read_parquet(path)


# =============================================================================
# Script de execução direta
# =============================================================================

if __name__ == "__main__":
    ingester = MarketDataIngester()

    # Exemplo: ações B3 + CDI
    tickers_br = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "WEGE3.SA"]
    precos = ingester.fetch_prices(tickers_br, "2020-01-01")
    retornos = ingester.calcular_retornos(precos)

    print(f"📊 Preços: {precos.shape}")
    print(f"📊 Retornos: {retornos.shape}")
    print(f"📊 Último retorno:\n{retornos.tail(1)}")
