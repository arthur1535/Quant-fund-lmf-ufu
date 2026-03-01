"""
QuantNucleo — Motor de Risco Quantitativo
============================================
Implementa: VaR, CVaR, Monte Carlo, drawdown, limites de posição,
GARCH vol forecasting, e validação de sinais.

Reaproveitado e evoluído de:
 - montecarlo repo: simulações MC
 - eua/analiseempresasamericanas.py: métricas de risco
 - LSTM-Previsão-Carteira: GARCH vol
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config.settings import RiskConfig, get_config
from libs.core.logging import get_logger
from libs.core.types import NivelRisco, Sinal

logger = get_logger("risk.engine")


class RiskEngine:
    """
    Motor de risco central do núcleo quant.
    
    Uso:
        engine = RiskEngine()
        var = engine.var_historico(retornos["PETR4.SA"])
        resultado_mc = engine.monte_carlo(retornos, pesos)
        aprovado = engine.validar_sinal(sinal)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or get_config().risk

    # =========================================================================
    # VaR (Value at Risk)
    # =========================================================================

    def var_historico(
        self,
        retornos: pd.Series,
        confianca: Optional[float] = None,
    ) -> float:
        """VaR histórico (percentil)."""
        conf = confianca or self.config.var_confidence
        return float(np.percentile(retornos.dropna(), (1 - conf) * 100))

    def var_parametrico(
        self,
        retornos: pd.Series,
        confianca: Optional[float] = None,
    ) -> float:
        """VaR paramétrico (Gaussiano)."""
        conf = confianca or self.config.var_confidence
        mu = retornos.mean()
        sigma = retornos.std()
        z = scipy_stats.norm.ppf(1 - conf)
        return float(mu + z * sigma)

    def cvar(
        self,
        retornos: pd.Series,
        confianca: Optional[float] = None,
    ) -> float:
        """Conditional VaR (Expected Shortfall)."""
        conf = confianca or self.config.cvar_confidence
        var = self.var_historico(retornos, conf)
        tail = retornos[retornos <= var]
        if tail.empty:
            return float(var)  # fallback: retorna o próprio VaR
        return float(tail.mean())

    # =========================================================================
    # Monte Carlo
    # =========================================================================

    def monte_carlo(
        self,
        retornos: pd.DataFrame,
        pesos: np.ndarray,
        capital_inicial: float = 10_000_000.0,
        simulacoes: Optional[int] = None,
        horizonte: Optional[int] = None,
        seed: int = 42,
    ) -> dict:
        """
        Simulação Monte Carlo para portfolio.
        
        Retorna:
            dict com valor_final_medio, var_95, cvar_95, paths, percentis
        """
        n_sim = simulacoes or self.config.mc_simulations
        n_dias = horizonte or self.config.mc_horizon_days

        np.random.seed(seed)

        retorno_portfolio = retornos.dot(pesos)
        mu = retorno_portfolio.mean()
        sigma = retorno_portfolio.std()

        # Gera caminhos simulados
        dt = 1  # 1 dia
        paths = np.zeros((n_sim, n_dias))
        paths[:, 0] = capital_inicial

        random_shocks = np.random.normal(
            mu * dt, sigma * np.sqrt(dt), (n_sim, n_dias - 1)
        )

        for t in range(1, n_dias):
            paths[:, t] = paths[:, t - 1] * (1 + random_shocks[:, t - 1])

        valores_finais = paths[:, -1]

        resultado = {
            "capital_inicial": capital_inicial,
            "simulacoes": n_sim,
            "horizonte_dias": n_dias,
            "valor_final_medio": float(np.mean(valores_finais)),
            "valor_final_mediana": float(np.median(valores_finais)),
            "valor_final_std": float(np.std(valores_finais)),
            "var_95": float(np.percentile(valores_finais, 5)),
            "cvar_95": float(np.mean(valores_finais[valores_finais <= np.percentile(valores_finais, 5)])),
            "percentis": {
                "p1": float(np.percentile(valores_finais, 1)),
                "p5": float(np.percentile(valores_finais, 5)),
                "p25": float(np.percentile(valores_finais, 25)),
                "p50": float(np.percentile(valores_finais, 50)),
                "p75": float(np.percentile(valores_finais, 75)),
                "p95": float(np.percentile(valores_finais, 95)),
                "p99": float(np.percentile(valores_finais, 99)),
            },
            "prob_perda": float(np.mean(valores_finais < capital_inicial)),
        }

        logger.info(
            "monte_carlo_concluido",
            simulacoes=n_sim,
            valor_medio=resultado["valor_final_medio"],
            var_95=resultado["var_95"],
        )
        return resultado

    # =========================================================================
    # Drawdown
    # =========================================================================

    def calcular_drawdown(self, retornos: pd.Series) -> pd.Series:
        """Calcula série de drawdown a partir de retornos log."""
        cumret = np.exp(retornos.cumsum())
        max_cumret = cumret.cummax()
        return cumret / max_cumret - 1

    def max_drawdown(self, retornos: pd.Series) -> float:
        """Drawdown máximo."""
        return float(self.calcular_drawdown(retornos).min())

    # =========================================================================
    # GARCH Volatility
    # =========================================================================

    def garch_vol(
        self,
        retornos: pd.Series,
        anualizar: int = 252,
    ) -> float:
        """
        GARCH(1,1) volatilidade prevista 1-step ahead (anualizada).
        Fallback → desvio padrão histórico.
        
        Reaproveitado de LSTM-Previsão-Carteira/portfolio_lstm_garch.py
        """
        r = (retornos - retornos.mean()).dropna()

        if len(r) < 180:
            logger.warning("garch_vol_fallback_hist", motivo="dados_insuficientes", n=len(r))
            return float(retornos.std() * np.sqrt(anualizar))

        try:
            from arch import arch_model

            am = arch_model(
                r * 100,
                p=self.config.garch_p,
                q=self.config.garch_q,
                mean="zero",
                vol="GARCH",
                dist=self.config.garch_dist,
            )
            res = am.fit(disp="off")
            fcast = res.forecast(horizon=1, reindex=False)
            sigma_daily_pct = np.sqrt(fcast.variance.values[-1, 0])
            sigma_daily = sigma_daily_pct / 100.0
            return float(sigma_daily * np.sqrt(anualizar))

        except Exception as e:
            logger.warning("garch_vol_fallback_erro", erro=str(e))
            return float(retornos.std() * np.sqrt(anualizar))

    # =========================================================================
    # Métricas de Performance
    # =========================================================================

    def sharpe_ratio(
        self,
        retornos: pd.Series,
        rf_anual: float = 0.0,
    ) -> float:
        """Sharpe ratio anualizado."""
        rf_diario = (1 + rf_anual) ** (1 / 252) - 1
        excess = retornos - rf_diario
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def sortino_ratio(
        self,
        retornos: pd.Series,
        rf_anual: float = 0.0,
    ) -> float:
        """Sortino ratio anualizado (downside deviation sobre todos os períodos)."""
        rf_diario = (1 + rf_anual) ** (1 / 252) - 1
        excess = retornos - rf_diario
        downside_diff = np.minimum(excess, 0)
        downside_dev = np.sqrt(np.mean(downside_diff ** 2))
        if downside_dev == 0:
            return float("inf")
        return float(excess.mean() / downside_dev * np.sqrt(252))

    def calmar_ratio(self, retornos: pd.Series) -> float:
        """Calmar ratio (retorno anualizado / max drawdown)."""
        ret_anual = retornos.mean() * 252
        mdd = abs(self.max_drawdown(retornos))
        if mdd == 0:
            return float("inf")
        return float(ret_anual / mdd)

    # =========================================================================
    # Validação de Sinais
    # =========================================================================

    def validar_sinal(self, sinal: Sinal) -> tuple[bool, list[str]]:
        """
        Valida sinal contra limites de risco do núcleo.
        
        Returns:
            (aprovado, lista_de_motivos_rejeicao)
        """
        rejeicoes: list[str] = []

        # 1. Peso máximo por ativo
        if sinal.peso_sugerido is not None and sinal.peso_sugerido > self.config.max_single_position:
            rejeicoes.append(
                f"Peso sugerido ({sinal.peso_sugerido:.1%}) > limite "
                f"({self.config.max_single_position:.1%})"
            )

        # 2. VaR estimado
        if sinal.var_estimado is not None and sinal.var_estimado < -0.10:
            rejeicoes.append(
                f"VaR estimado ({sinal.var_estimado:.1%}) > threshold (-10%)"
            )

        # 3. Drawdown
        if sinal.drawdown_maximo is not None and sinal.drawdown_maximo < self.config.max_drawdown_threshold:
            rejeicoes.append(
                f"Drawdown ({sinal.drawdown_maximo:.1%}) > threshold "
                f"({self.config.max_drawdown_threshold:.1%})"
            )

        # 4. Confiança mínima
        if sinal.confianca < 0.3:
            rejeicoes.append(
                f"Confiança muito baixa ({sinal.confianca:.1%})"
            )

        # 5. Nível de risco
        if sinal.nivel_risco == NivelRisco.CRITICO:
            rejeicoes.append("Sinal com nível de risco CRITICO requer revisão manual")

        aprovado = len(rejeicoes) == 0
        logger.info(
            "validacao_sinal",
            sinal_id=str(sinal.id),
            ativo=sinal.ativo.ticker,
            tipo=sinal.tipo.value,
            aprovado=aprovado,
            rejeicoes=rejeicoes,
        )
        return aprovado, rejeicoes

    # =========================================================================
    # Relatório de Risco
    # =========================================================================

    def relatorio_risco(
        self,
        retornos: pd.DataFrame,
        pesos: np.ndarray,
        rf_anual: float = 0.0,
    ) -> dict:
        """Gera relatório de risco completo do portfolio."""
        retorno_portfolio = retornos.dot(pesos)

        return {
            "sharpe": self.sharpe_ratio(retorno_portfolio, rf_anual),
            "sortino": self.sortino_ratio(retorno_portfolio, rf_anual),
            "calmar": self.calmar_ratio(retorno_portfolio),
            "max_drawdown": self.max_drawdown(retorno_portfolio),
            "var_95_historico": self.var_historico(retorno_portfolio),
            "var_95_parametrico": self.var_parametrico(retorno_portfolio),
            "cvar_95": self.cvar(retorno_portfolio),
            "volatilidade_anual": float(retorno_portfolio.std() * np.sqrt(252)),
            "retorno_anual": float(retorno_portfolio.mean() * 252),
            "garch_vol": self.garch_vol(retorno_portfolio),
            "por_ativo": {
                col: {
                    "var_95": self.var_historico(retornos[col]),
                    "max_drawdown": self.max_drawdown(retornos[col]),
                    "vol_anual": float(retornos[col].std() * np.sqrt(252)),
                    "peso": float(pesos[i]),
                }
                for i, col in enumerate(retornos.columns)
            },
        }

    # =========================================================================
    # Métodos Avançados de Risco
    # =========================================================================

    def var_cornish_fisher(
        self,
        retornos: pd.Series,
        confianca: float = 0.95,
    ) -> float:
        """
        VaR ajustado por Cornish-Fisher (expansão para caudas pesadas).

        Ajusta o quantil normal usando skewness e kurtosis da distribuição
        real dos retornos, capturando melhor o risco de cauda.
        """
        from scipy.stats import norm

        z = norm.ppf(1 - confianca)
        s = float(retornos.skew())
        k = float(retornos.kurtosis())

        z_cf = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36
        )

        var = float(retornos.mean() + z_cf * retornos.std())

        logger.info("var_cornish_fisher", var=round(var, 6), skew=round(s, 4), kurtosis=round(k, 4))
        return var

    def stress_test(
        self,
        retornos: pd.DataFrame,
        pesos: np.ndarray,
        cenarios: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """
        Stress test do portfolio sob cenários definidos.

        Args:
            retornos: DataFrame de retornos dos ativos
            pesos: Pesos do portfolio
            cenarios: Dict de {nome_cenario: array de retornos por ativo}

        Returns:
            Dict de {nome_cenario: retorno_portfolio sob estresse}
        """
        resultados = {}
        for nome, retornos_cenario in cenarios.items():
            retorno = float(np.dot(pesos, retornos_cenario))
            resultados[nome] = retorno
            logger.info("stress_test", cenario=nome, retorno=round(retorno, 6))

        return resultados

    def tracking_error(
        self,
        retornos_portfolio: pd.Series,
        retornos_benchmark: pd.Series,
    ) -> float:
        """
        Tracking Error anualizado (volatilidade do excesso de retorno).

        TE = std(Rp - Rb) × √252
        """
        excesso = retornos_portfolio - retornos_benchmark
        te = float(excesso.std() * np.sqrt(252))
        logger.info("tracking_error", te=round(te, 6))
        return te

    def information_ratio(
        self,
        retornos_portfolio: pd.Series,
        retornos_benchmark: pd.Series,
    ) -> float:
        """
        Information Ratio — retorno ativo por unidade de tracking error.

        IR = (Rp_anual - Rb_anual) / TE
        """
        excesso = retornos_portfolio - retornos_benchmark
        te = excesso.std() * np.sqrt(252)
        if te == 0:
            return 0.0
        ir = float(excesso.mean() * 252 / te)
        logger.info("information_ratio", ir=round(ir, 4))
        return ir

    def tail_ratio(self, retornos: pd.Series) -> float:
        """
        Tail Ratio — razão entre os percentis extremos.

        Tail Ratio = |percentil 95| / |percentil 5|

        Mede assimetria das caudas. > 1 indica cauda positiva maior.
        """
        p95 = np.percentile(retornos.dropna(), 95)
        p5 = np.percentile(retornos.dropna(), 5)
        if abs(p5) == 0:
            return 0.0
        return float(abs(p95) / abs(p5))
