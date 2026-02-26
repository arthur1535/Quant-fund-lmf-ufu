# =====================================================================
# QuantNucleo — Research: LSTM Integration
# =====================================================================
# Wrapper para integrar modelos LSTM existentes no pipeline do QuantNucleo.
# Reutiliza a arquitetura de LSTM-Previsão-de-Carteira-main/portfolio_lstm_garch.py.

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from dataclasses import dataclass, field

logger = structlog.get_logger(__name__)


@dataclass
class LSTMConfig:
    """Configuração do modelo LSTM."""
    lookback: int = 60
    horizonte: int = 30
    unidades_lstm: int = 50
    dropout: float = 0.2
    epocas: int = 100
    batch_size: int = 32
    patience: int = 5
    feature_range: tuple[int, int] = (0, 1)


@dataclass
class PrevisaoLSTM:
    """Resultado de uma previsão LSTM."""
    ticker: str
    previsoes: np.ndarray
    datas_previstas: list[str]
    mse: float
    rmse: float
    intervalo_confianca: tuple[np.ndarray, np.ndarray] | None = None
    historico_loss: list[float] = field(default_factory=list)


class LSTMForecaster:
    """
    Forecaster LSTM para séries de preços.

    Arquitetura padrão (conforme copilot-instructions.md):
        Sequential([
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
    """

    def __init__(self, config: LSTMConfig | None = None):
        self.config = config or LSTMConfig()
        self.modelo = None
        self.scaler = None
        self._importar_dependencias()

    def _importar_dependencias(self):
        """Import lazy para não exigir TensorFlow como dependência obrigatória."""
        try:
            from sklearn.preprocessing import MinMaxScaler
            self._MinMaxScaler = MinMaxScaler
            logger.info("✅ sklearn disponível para LSTM")
        except ImportError:
            logger.warning("⚠️ sklearn não disponível")
            self._MinMaxScaler = None

    def _construir_modelo(self, input_shape: tuple):
        """Constrói a arquitetura LSTM padrão."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dropout, Dense
            from tensorflow.keras.callbacks import EarlyStopping

            modelo = Sequential([
                LSTM(self.config.unidades_lstm, return_sequences=True,
                     input_shape=input_shape),
                Dropout(self.config.dropout),
                LSTM(self.config.unidades_lstm),
                Dropout(self.config.dropout),
                Dense(1),
            ])
            modelo.compile(optimizer="adam", loss="mse")
            self.modelo = modelo
            self._EarlyStopping = EarlyStopping

            logger.info(
                "🧠 Modelo LSTM construído",
                input_shape=input_shape,
                parametros=modelo.count_params(),
            )
            return modelo

        except ImportError:
            logger.error("❌ TensorFlow não instalado. Use: pip install tensorflow")
            raise

    def _preparar_dados(
        self, precos: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara dados para treinamento LSTM.

        Returns: (X_train, y_train, dados_escalados)
        """
        if self._MinMaxScaler is None:
            raise ImportError("sklearn necessário")

        self.scaler = self._MinMaxScaler(
            feature_range=self.config.feature_range
        )
        dados = self.scaler.fit_transform(
            precos.values.reshape(-1, 1)
        )

        X, y = [], []
        for i in range(self.config.lookback, len(dados)):
            X.append(dados[i - self.config.lookback : i, 0])
            y.append(dados[i, 0])

        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y, dados

    def treinar(self, precos: pd.Series) -> dict:
        """
        Treina modelo LSTM com série de preços.

        Args:
            precos: pd.Series com preços de fechamento (index=datas).

        Returns: Métricas de treinamento
        """
        logger.info(
            "🏋️ Iniciando treinamento LSTM",
            pontos=len(precos),
            lookback=self.config.lookback,
        )

        X, y, dados = self._preparar_dados(precos)

        # Split treino/teste (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Construir modelo
        self._construir_modelo(input_shape=(X_train.shape[1], 1))

        # Treinar
        early_stop = self._EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience,
            restore_best_weights=True,
        )

        historico = self.modelo.fit(
            X_train, y_train,
            epochs=self.config.epocas,
            batch_size=self.config.batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0,
        )

        # Métricas
        previsoes_test = self.modelo.predict(X_test, verbose=0)
        previsoes_inv = self.scaler.inverse_transform(previsoes_test)
        real_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = float(np.mean((previsoes_inv - real_inv) ** 2))
        rmse = float(np.sqrt(mse))

        metricas = {
            "mse": mse,
            "rmse": rmse,
            "epocas_executadas": len(historico.history["loss"]),
            "loss_final": float(historico.history["loss"][-1]),
            "val_loss_final": float(historico.history["val_loss"][-1]),
        }

        logger.info("✅ Treinamento completo", **metricas)
        return metricas

    def prever(
        self,
        precos: pd.Series,
        ticker: str = "UNKNOWN",
    ) -> PrevisaoLSTM:
        """
        Gera previsões para os próximos N dias.

        Args:
            precos: Série de preços históricos.
            ticker: Nome do ativo.

        Returns: PrevisaoLSTM com resultados.
        """
        if self.modelo is None:
            raise RuntimeError("Modelo não treinado. Chame treinar() primeiro.")

        _, _, dados = self._preparar_dados(precos)

        # Usar últimos `lookback` pontos como seed
        seed = dados[-self.config.lookback :].reshape(1, self.config.lookback, 1)

        previsoes = []
        entrada_atual = seed.copy()

        for _ in range(self.config.horizonte):
            pred = self.modelo.predict(entrada_atual, verbose=0)
            previsoes.append(pred[0, 0])
            # Shift: remove primeiro, adiciona previsão
            entrada_atual = np.append(
                entrada_atual[:, 1:, :],
                pred.reshape(1, 1, 1),
                axis=1,
            )

        previsoes_arr = np.array(previsoes).reshape(-1, 1)
        previsoes_reais = self.scaler.inverse_transform(previsoes_arr).flatten()

        # Datas futuras
        ultima_data = precos.index[-1]
        datas_futuras = pd.bdate_range(
            start=ultima_data + pd.Timedelta(days=1),
            periods=self.config.horizonte,
        )

        # MSE placeholder (será avaliado a posteriori)
        resultado = PrevisaoLSTM(
            ticker=ticker,
            previsoes=previsoes_reais,
            datas_previstas=[d.strftime("%Y-%m-%d") for d in datas_futuras],
            mse=0.0,
            rmse=0.0,
        )

        logger.info(
            "📈 Previsão gerada",
            ticker=ticker,
            horizonte=self.config.horizonte,
            preco_atual=float(precos.iloc[-1]),
            preco_previsto_final=float(previsoes_reais[-1]),
        )

        return resultado

    def prever_multiplos(
        self, precos_dict: dict[str, pd.Series]
    ) -> dict[str, PrevisaoLSTM]:
        """
        Treina e prevê para múltiplos ativos.

        Args:
            precos_dict: {ticker: pd.Series de preços}

        Returns: {ticker: PrevisaoLSTM}
        """
        resultados = {}
        for ticker, precos in precos_dict.items():
            logger.info(f"🔄 Processando {ticker}...")
            try:
                self.treinar(precos)
                resultado = self.prever(precos, ticker=ticker)
                resultados[ticker] = resultado
            except Exception as e:
                logger.error(f"❌ Erro no LSTM para {ticker}: {e}")
        return resultados
