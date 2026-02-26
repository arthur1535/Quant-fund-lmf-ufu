"""
QuantNucleo — Logging Estruturado
==================================
Usa structlog para logs JSON em produção, colorido em dev.
Integra com Datadog (ddtrace) e Sentry automaticamente.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from config.settings import Ambiente, get_config


def _setup_structlog() -> None:
    """Configura structlog uma vez."""
    cfg = get_config()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if cfg.ambiente == Ambiente.PROD:
        # Produção: JSON para Datadog
        renderer = structlog.processors.JSONRenderer()
    else:
        # Dev: colorido e legível
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))


# Inicializa na importação
_setup_structlog()


def get_logger(name: str = "quantnucleo") -> structlog.stdlib.BoundLogger:
    """Retorna logger estruturado vinculado ao nome do módulo."""
    return structlog.get_logger(name)
