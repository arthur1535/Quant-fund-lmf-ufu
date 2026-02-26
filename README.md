# 🏛️ QuantNucleo — Núcleo Quantitativo LMF/UFU

[![CI/CD](https://github.com/arthur1535/Quant-fund-lmf.ufu/actions/workflows/ci.yml/badge.svg)](https://github.com/arthur1535/Quant-fund-lmf.ufu/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Visão Geral

Plataforma central de pesquisa quantitativa, backtesting institucional, gestão de risco e geração sistemática de sinais para o Núcleo Quant LMF/UFU.

**Não** é apenas trading automatizado — é uma **infraestrutura profissional** para:

- 📊 Pesquisa quantitativa contínua e reprodutível
- 🧪 Backtesting institucional auditável
- 📡 Geração sistemática de sinais
- 🛡️ Gestão de risco quantitativo
- 🔄 Pipelines reprodutíveis (LSTM, GARCH, Monte Carlo)
- ⚡ Alertas em tempo real via TradingView + execução assistida
- 🚀 Evolução para execução automatizada (após maturidade)

## Stack Tecnológico

| Camada | Tecnologia |
|--------|-----------|
| **Core** | Python (Polars, NumPy, VectorBT, PyTorch) |
| **Execution Engine** | Rust (latência crítica) |
| **Data Lake** | Azure Blob Storage → Synapse Analytics |
| **Banco Operacional** | MongoDB Atlas |
| **Orquestração** | Azure Kubernetes Service (AKS) |
| **Segurança** | Azure Key Vault + 1Password |
| **CI/CD** | GitHub Actions |
| **Observabilidade** | Datadog + Sentry |
| **Alertas** | TradingView Webhooks → Azure Functions |
| **Dev Environment** | GitHub Codespaces + Copilot |

## Estrutura do Monorepo

```
quantnucleo/
├── .github/                    # CI/CD, templates, CODEOWNERS
│   └── workflows/
├── config/                     # Configurações centrais
├── data/                       # Data lakehouse (local dev)
│   ├── raw/                    # Dados brutos ingestados
│   ├── processed/              # Dados processados
│   └── features/               # Feature store
├── docs/                       # Documentação arquitetural
├── infra/                      # IaC (Terraform, Bicep, Docker, K8s)
│   ├── docker/
│   ├── k8s/
│   └── terraform/
├── libs/                       # Bibliotecas compartilhadas
│   ├── core/                   # Utilidades base (logging, config, types)
│   ├── data/                   # Ingestão, transformação, feature store
│   ├── risk/                   # Motor de risco (VaR, Monte Carlo, drawdown)
│   └── signals/                # Geração e validação de sinais
├── research/                   # Pesquisa (notebooks, experimentos)
│   ├── forecasting/            # LSTM, GARCH, modelos preditivos
│   ├── montecarlo/             # Simulações de risco
│   └── energy/                 # Pipeline energia quantitativa
├── services/                   # Serviços de produção
│   ├── api/                    # API REST principal
│   ├── webhook_receiver/       # Receptor TradingView webhooks
│   ├── signal_engine/          # Motor de sinais
│   ├── risk_gateway/           # Gateway de risco
│   └── backtest_runner/        # Executor de backtests
├── strategies/                 # Estratégias de trading
│   ├── base.py                 # Classe abstrata de estratégia
│   └── examples/               # Exemplos/templates
├── tests/                      # Testes (unit, integration, backtest)
│   ├── unit/
│   ├── integration/
│   └── backtest/
├── tradingview/                # Pine Scripts + config webhooks
│   ├── pinescripts/
│   └── webhook_config/
├── pyproject.toml              # Config Python monorepo
├── Makefile                    # Comandos de conveniência
└── docker-compose.yml          # Dev environment local
```

## Início Rápido

```bash
# 1. Clonar
git clone https://github.com/arthur1535/Quant-fund-lmf.ufu.git
cd Quant-fund-lmf.ufu

# 2. Setup ambiente (Codespaces ou local)
make setup

# 3. Rodar testes
make test

# 4. Executar backtest exemplo
make backtest-smoke

# 5. Iniciar serviços localmente
docker-compose up -d
```

## Repositórios Integrados (Subtrees)

| Subtree Path | Repositório Origem | Papel |
|---|---|---|
| `research/montecarlo/` | [montecarlo](https://github.com/arthur1535/montecarlo.git) | Simulações de risco |
| `research/forecasting/lstm/` | [LSTM](https://github.com/arthur1535/LSTM.git) | Modelos LSTM base |
| `research/forecasting/portfolio/` | [LSTM-Previsão-Carteira](https://github.com/arthur1535/LSTM-Previs-o-de-Carteira.git) | LSTM+GARCH+Markowitz |
| `research/energy/` | [quantitative-energy-thesis](https://github.com/arthur1535/quantitative-energy-thesis.git) | Pipeline energia quant |

## Licença

MIT — Veja [LICENSE](LICENSE) para detalhes.
