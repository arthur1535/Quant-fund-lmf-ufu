"""
QuantNucleo — Feature Store Quantitativo
==========================================
Computa e armazena features quantitativas derivadas dos dados brutos.

Features implementadas:
 - Retornos (log, simples)
 - Volatilidade (rolling, EWMA, GARCH)
 - Momentum (RSI, MACD, ROC)
 - Drawdown (max drawdown, current drawdown)
 - Correlação (rolling, cross-asset)
 - Volume (normalizado, OBV)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from libs.core.logging import get_logger

logger = get_logger("data.features")


class FeatureStore:
    """
    Computa features quantitativas para uso em modelos e sinais.
    
    Uso:
        fs = FeatureStore(precos)
        features = fs.compute_all()
    """

    def __init__(self, precos: pd.DataFrame) -> None:
        """
        Args:
            precos: DataFrame com preços de fechamento (coluna por ativo).
        """
        self.precos = precos.ffill().dropna()
        self.retornos_log = np.log(self.precos / self.precos.shift(1)).dropna()
        self.retornos_simples = self.precos.pct_change().dropna()

    def volatilidade_rolling(self, janela: int = 21) -> pd.DataFrame:
        """Volatilidade rolling anualizada (√252)."""
        return self.retornos_log.rolling(janela).std() * np.sqrt(252)

    def volatilidade_ewma(self, span: int = 21) -> pd.DataFrame:
        """Volatilidade EWMA anualizada."""
        return self.retornos_log.ewm(span=span).std() * np.sqrt(252)

    def rsi(self, janela: int = 14) -> pd.DataFrame:
        """Relative Strength Index (RSI)."""
        delta = self.precos.diff()
        ganhos = delta.where(delta > 0, 0.0).rolling(janela).mean()
        perdas = (-delta.where(delta < 0, 0.0)).rolling(janela).mean()
        rs = ganhos / perdas.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def macd(
        self,
        rapida: int = 12,
        lenta: int = 26,
        sinal: int = 9,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """MACD, sinal e histograma."""
        ema_rapida = self.precos.ewm(span=rapida).mean()
        ema_lenta = self.precos.ewm(span=lenta).mean()
        macd_line = ema_rapida - ema_lenta
        signal_line = macd_line.ewm(span=sinal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def drawdown(self) -> pd.DataFrame:
        """Drawdown corrente (base log-returns)."""
        cumret = self.retornos_log.cumsum().apply(np.exp)
        max_cumret = cumret.cummax()
        return cumret / max_cumret - 1

    def max_drawdown(self) -> pd.Series:
        """Drawdown máximo por ativo."""
        return self.drawdown().min()

    def sharpe_rolling(
        self,
        janela: int = 63,
        rf_diario: float = 0.0,
    ) -> pd.DataFrame:
        """Sharpe ratio rolling (janela em dias úteis)."""
        ret_medio = self.retornos_log.rolling(janela).mean()
        vol = self.retornos_log.rolling(janela).std()
        return (ret_medio - rf_diario) / vol * np.sqrt(252)

    def sortino_rolling(
        self,
        janela: int = 63,
        rf_diario: float = 0.0,
    ) -> pd.DataFrame:
        """Sortino ratio rolling."""
        ret_medio = self.retornos_log.rolling(janela).mean()
        neg_ret = self.retornos_log.where(self.retornos_log < 0, 0.0)
        downside_vol = neg_ret.rolling(janela).std()
        return (ret_medio - rf_diario) / downside_vol * np.sqrt(252)

    def correlacao_rolling(self, janela: int = 63) -> dict[str, pd.DataFrame]:
        """Matriz de correlação rolling entre ativos."""
        result = {}
        for col in self.retornos_log.columns:
            corr_df = self.retornos_log.rolling(janela).corr(
                self.retornos_log[col]
            )
            # Garantir que o resultado é um DataFrame simples (sem MultiIndex)
            if isinstance(corr_df.index, pd.MultiIndex):
                corr_df = corr_df.droplevel(1)
            result[col] = corr_df
        return result

    # =========================================================================
    # Features de Renda Fixa
    # =========================================================================

    def spread_cdi_rolling(
        self,
        retornos_ativo: pd.Series,
        cdi_diario: pd.Series,
        janela: int = 21,
    ) -> pd.Series:
        """Spread do retorno do ativo sobre o CDI (rolling)."""
        excesso = retornos_ativo - cdi_diario
        return excesso.rolling(janela).mean() * 252

    def taxa_real_implica(
        self,
        taxa_nominal: pd.Series,
        ipca_esperado: float = 0.045,
    ) -> pd.Series:
        """Taxa real implícita: (1+nominal)/(1+inflação) - 1."""
        return (1 + taxa_nominal) / (1 + ipca_esperado) - 1

    def carry_trade_signal(
        self,
        taxa_curta: pd.Series,
        taxa_longa: pd.Series,
    ) -> pd.Series:
        """
        Sinal de carry trade: diferença entre taxa longa e curta.
        Positivo = curva inclinada (favorece posição tomada na ponta longa).
        """
        return taxa_longa - taxa_curta

    def retorno_acumulado_cdi(
        self,
        cdi_diario: pd.Series,
    ) -> pd.Series:
        """Retorno acumulado do CDI (benchmark de renda fixa)."""
        return (1 + cdi_diario).cumprod() - 1

    # =========================================================================
    # Features de Valuation
    # =========================================================================

    def roc(self, janela: int = 21) -> pd.DataFrame:
        """Rate of Change (momentum)."""
        return self.precos / self.precos.shift(janela) - 1

    def bollinger_bands(
        self,
        janela: int = 20,
        num_std: float = 2.0,
    ) -> dict[str, pd.DataFrame]:
        """Bandas de Bollinger (média, superior, inferior, %B)."""
        media = self.precos.rolling(janela).mean()
        std = self.precos.rolling(janela).std()
        superior = media + num_std * std
        inferior = media - num_std * std
        pct_b = (self.precos - inferior) / (superior - inferior)
        return {
            "media": media,
            "superior": superior,
            "inferior": inferior,
            "pct_b": pct_b,
        }

    def z_score_preco(self, janela: int = 63) -> pd.DataFrame:
        """Z-score do preço (desvios da média rolling)."""
        media = self.precos.rolling(janela).mean()
        std = self.precos.rolling(janela).std()
        return (self.precos - media) / std

    def obv(self, volumes: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume (OBV)."""
        sinais = np.sign(self.precos.diff())
        obv_values = (sinais * volumes).cumsum()
        return obv_values

    def compute_all(self, janela_vol: int = 21) -> pd.DataFrame:
        """Computa todas as features e retorna DataFrame unificado."""
        features = pd.DataFrame(index=self.precos.index)

        for col in self.precos.columns:
            prefix = col.replace(".", "_").replace("=", "_")
            features[f"{prefix}_ret_log"] = self.retornos_log.get(col)
            features[f"{prefix}_ret_simples"] = self.retornos_simples.get(col)
            features[f"{prefix}_vol_rolling"] = self.volatilidade_rolling(janela_vol).get(col)
            features[f"{prefix}_vol_ewma"] = self.volatilidade_ewma(janela_vol).get(col)
            features[f"{prefix}_rsi"] = self.rsi().get(col)
            features[f"{prefix}_drawdown"] = self.drawdown().get(col)
            features[f"{prefix}_sharpe_63d"] = self.sharpe_rolling(63).get(col)
            features[f"{prefix}_roc_21d"] = self.roc(21).get(col)
            features[f"{prefix}_z_score_63d"] = self.z_score_preco(63).get(col)

            # Bandas de Bollinger
            bb = self.bollinger_bands()
            features[f"{prefix}_bb_pct_b"] = bb["pct_b"].get(col)

        logger.info("features_computadas", total_colunas=len(features.columns))
        return features.dropna()
