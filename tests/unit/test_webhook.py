"""
QuantNucleo — Testes Unitários: Webhook Receiver
==================================================
"""

import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from services.webhook_receiver.main import app


class TestWebhookReceiver(unittest.TestCase):
    """Testes do receptor de webhooks TradingView."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_check(self) -> None:
        """Health check deve retornar 200."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "webhook_receiver")

    def test_webhook_valido(self) -> None:
        """Webhook com payload válido deve retornar 202."""
        payload = {
            "ticker": "PETR4.SA",
            "action": "buy",
            "price": 38.50,
            "interval": "1D",
            "exchange": "BMFBOVESPA",
            "strategy_name": "test_strategy",
            "message": "Teste webhook",
        }
        response = self.client.post(
            "/api/v1/webhooks/tradingview",
            json=payload,
        )
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["status"], "aceito")
        self.assertIsNotNone(data["signal_id"])

    def test_webhook_sell(self) -> None:
        """Webhook de venda deve funcionar."""
        payload = {
            "ticker": "VALE3.SA",
            "action": "sell",
            "price": 62.30,
            "message": "Death Cross",
        }
        response = self.client.post(
            "/api/v1/webhooks/tradingview",
            json=payload,
        )
        self.assertEqual(response.status_code, 202)

    def test_webhook_idempotencia(self) -> None:
        """Webhooks duplicados devem ser detectados."""
        payload = {
            "ticker": "PETR4.SA",
            "action": "buy",
            "price": 38.50,
            "idempotency_key": "test_dedup_key_12345",
        }
        # Primeiro envio
        r1 = self.client.post("/api/v1/webhooks/tradingview", json=payload)
        self.assertEqual(r1.status_code, 202)
        self.assertEqual(r1.json()["status"], "aceito")

        # Segundo envio (mesmo idempotency key)
        r2 = self.client.post("/api/v1/webhooks/tradingview", json=payload)
        self.assertEqual(r2.status_code, 202)
        self.assertEqual(r2.json()["status"], "duplicado")

    def test_sinais_pendentes_vazio(self) -> None:
        """Lista de sinais pendentes deve funcionar."""
        response = self.client.get("/api/v1/signals/pending")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_webhook_payload_invalido(self) -> None:
        """Payload inválido deve retornar 422."""
        response = self.client.post(
            "/api/v1/webhooks/tradingview",
            json={"invalido": True},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
