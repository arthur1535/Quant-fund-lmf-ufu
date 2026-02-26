# =====================================================================
# QuantNucleo — Arquitetura Técnica Completa
# =====================================================================

## 1. Visão Geral

O **QuantNucleo** é a plataforma de infraestrutura quantitativa do Núcleo Quant do **LMF/UFU**,
projetada como um monorepo institucional que integra pesquisa, backtesting, geração de sinais,
gestão de risco e execução — tudo com governança, auditoria e observabilidade completas.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRADINGVIEW (Pine Script)                   │
│               alertcondition() → JSON Payload                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS POST (webhook)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              WEBHOOK RECEIVER (FastAPI :8001)                   │
│   Token auth + IP allowlist + Idempotency + Audit               │
└────────────────────────┬────────────────────────────────────────┘
                         │ Sinal validado
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL ENGINE                                │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐                   │
│  │Momentum │  │   RSI    │  │ TradingView │  (fontes)           │
│  └────┬────┘  └────┬─────┘  └──────┬──────┘                    │
│       └────────────┼───────────────┘                            │
│                    ▼                                             │
│           validar_e_registrar()                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RISK ENGINE                                 │
│  ┌─────┐ ┌──────┐ ┌───────────┐ ┌───────┐ ┌────────────────┐  │
│  │ VaR │ │ CVaR │ │Monte Carlo│ │ GARCH │ │Signal Validator│  │
│  └─────┘ └──────┘ └───────────┘ └───────┘ └────────────────┘  │
│                                                                 │
│  Regras: peso_max=25%, var_threshold=-5%, drawdown_max=-15%     │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     APROVADO    PENDENTE    REJEITADO
     (auto)    (human-in-   (bloqueado)
               the-loop)
```

## 2. Estrutura do Monorepo

```
quantnucleo/
├── config/                 # Configuração centralizada (Pydantic)
├── libs/
│   ├── core/               # Logging, types, CLI
│   ├── data/               # Ingestão, feature store
│   ├── risk/               # Motor de risco
│   ├── signals/            # Motor de sinais
│   └── monitoring/         # Métricas e observabilidade
├── services/
│   ├── api/                # API REST principal (:8000)
│   ├── webhook_receiver/   # Receptor TradingView (:8001)
│   ├── signal_engine/      # Processador de sinais
│   ├── risk_gateway/       # Gateway de risco
│   └── backtest_runner/    # Executor de backtests
├── strategies/             # Estratégias quantitativas
│   ├── base.py             # Classe abstrata base
│   └── examples/           # Implementações de referência
├── tradingview/            # Pine Scripts + config webhooks
├── research/               # Pesquisa e modelos ML
│   ├── forecasting/        # LSTM, GARCH
│   └── notebooks/          # Jupyter exploratório
├── infra/
│   ├── docker/             # Dockerfiles
│   ├── k8s/                # Manifests Kubernetes
│   └── terraform/          # IaC Azure
├── tests/
│   ├── unit/               # Testes unitários
│   ├── integration/        # Testes integração
│   └── backtest/           # Smoke tests backtest
└── docs/                   # Documentação
```

## 3. Data Lakehouse Architecture

```
Azure Blob Storage (Data Lake Gen2)
├── raw/                    # Dados brutos imutáveis
│   ├── market/             # Preços OHLCV (yfinance)
│   ├── bcb/                # Séries BCB (Selic, CDI, IPCA)
│   ├── cvm/                # Dados CVM (fundos)
│   └── webhooks/           # Payloads TradingView (audit)
├── processed/              # Dados limpos e transformados
│   ├── returns/            # Retornos diários (log/simples)
│   ├── covariance/         # Matrizes covariância
│   └── signals/            # Sinais processados
└── features/               # Feature Store
    ├── volatility/         # Volatilidade (rolling, EWMA, GARCH)
    ├── momentum/           # RSI, MACD, MMs
    ├── risk/               # VaR, CVaR, drawdown
    └── portfolio/          # Métricas de portfólio
```

**Formato**: Apache Parquet (compressão snappy)
**Particionamento**: `year=YYYY/month=MM/ticker=XXXX`
**Retenção**: Raw=indefinido, Processed=2 anos, Features=1 ano

## 4. Fluxo de Eventos: TradingView → Execução

```
1. ALERTA TRADINGVIEW
   └─▶ Pine Script gera JSON com ticker, preço, tipo, confiança
   
2. WEBHOOK RECEIVER
   ├─▶ Verifica token X-Webhook-Token
   ├─▶ Verifica IP allowlist (52.89.214.238, 34.212.75.30, etc.)
   ├─▶ Verifica idempotency (SHA256 do payload)
   ├─▶ Cria objeto Sinal com UUID e timestamps
   └─▶ Envia para Signal Engine

3. SIGNAL ENGINE
   ├─▶ Enriquece sinal com métricas (RSI, volatilidade)
   ├─▶ Define peso sugerido baseado em confiança
   └─▶ Encaminha para Risk Engine

4. RISK ENGINE (validação)
   ├─▶ Verifica peso máximo (25%)
   ├─▶ Verifica VaR marginal
   ├─▶ Verifica drawdown do portfólio
   ├─▶ Verifica confiança mínima (0.3)
   └─▶ Decide: APROVADO / PENDENTE / REJEITADO

5. EXECUÇÃO
   ├─▶ APROVADO → Log em MongoDB + métricas Datadog + notificação
   ├─▶ PENDENTE → Fila para aprovação humana (Slack/email)
   └─▶ REJEITADO → Log de auditoria + motivo detalhado

6. AUDITORIA
   └─▶ Todo sinal gera registro imutável em MongoDB
       com: timestamp, origem, decisão, métricas, quem aprovou
```

## 5. Stack Tecnológico

| Componente | Tecnologia | Justificativa |
|-----------|-----------|--------------|
| Linguagem | Python 3.11 | Ecossistema quant maduro |
| API | FastAPI + Uvicorn | Async, tipado, OpenAPI auto |
| Dados | Polars + Pandas | Performance + compatibilidade |
| ML/DL | PyTorch, TensorFlow/Keras | LSTM, modelos avançados |
| Risco | NumPy, SciPy, arch | VaR, Monte Carlo, GARCH |
| Otimização | PyPortfolioOpt | Markowitz + fronteira eficiente |
| Config | Pydantic v2 + pydantic-settings | Validação forte, SecretStr |
| Logging | structlog | JSON estruturado, Datadog-ready |
| Observabilidade | Datadog + Sentry | Métricas + rastreamento erros |
| Infra | Azure AKS, Blob, Key Vault | Enterprise, escalável |
| IaC | Terraform | Reproducibilidade infra |
| CI/CD | GitHub Actions | Integração nativa com repo |
| Contêiner | Docker | Builds reproduzíveis |
| Orquestração | Kubernetes (AKS) | Auto-scaling, rolling deploys |
| Banco | MongoDB Atlas | Flexível para sinais/auditoria |

## 6. Segurança (Zero-Trust)

### 6.1 Camadas de Segurança
```
┌──────────────────────────────────────┐
│         Azure Key Vault              │
│   (todos os segredos centralizados)  │
├──────────────────────────────────────┤
│  - MONGO_URI                         │
│  - TV_WEBHOOK_SECRET                 │
│  - DATADOG_API_KEY                   │
│  - SENTRY_DSN                        │
│  - AZURE_STORAGE_KEY                 │
└──────────────────────────────────────┘
```

### 6.2 Princípios
- **Never hardcode secrets**: Tudo via env vars ou Key Vault
- **SecretStr no Pydantic**: Segredos nunca aparecem em logs/repr
- **IP Allowlist**: Webhook aceita apenas IPs do TradingView
- **Token rotation**: Tokens webhook rodam a cada 90 dias
- **Audit trail**: Todo acesso/decisão registrado com timestamp
- **RBAC**: Kubernetes service accounts com permissões mínimas
- **Network policies**: Calico no AKS para microsegmentação

## 7. Métricas e Alertas

### 7.1 Métricas Quantitativas (Datadog)
| Métrica | Tipo | Alerta |
|---------|------|--------|
| `quantnucleo.risk.var_95` | gauge | < -5% |
| `quantnucleo.risk.drawdown_max` | gauge | < -15% |
| `quantnucleo.risk.sharpe_rolling` | gauge | < 0 (30d) |
| `quantnucleo.risk.volatilidade_garch` | gauge | — |
| `quantnucleo.signals.pendentes` | gauge | > 10 |
| `quantnucleo.webhook.latencia` | histogram | p99 > 500ms |
| `quantnucleo.infra.heartbeat` | gauge | < 1 |

### 7.2 Dashboards
1. **Painel Risco**: VaR/CVaR live, drawdown, correlação, volatilidade GARCH
2. **Painel Sinais**: Fluxo de sinais (recebidos → aprovados → rejeitados)
3. **Painel Infra**: Latência API, CPU/mem AKS, webhook throughput
4. **Painel LSTM**: MSE treino, desvio previsão, accuracy tracking

---

## 8. Roadmap

### Fase 1 — Fundação (30 dias)
- [x] Estrutura monorepo
- [x] Core libs (logging, types, CLI)
- [x] Risk Engine (VaR, CVaR, Monte Carlo, GARCH)
- [x] Signal Engine + Webhook Receiver
- [x] CI/CD pipeline completo
- [x] Testes unitários + smoke
- [x] Docker + K8s manifests
- [x] Terraform base (AKS, Blob, Key Vault)
- [x] Estratégia de referência (Momentum Crossover)
- [x] Pine Script base TradingView
- [ ] Deploy inicial no AKS (staging)
- [ ] Conectar MongoDB Atlas
- [ ] Configurar Datadog agent no AKS

### Fase 2 — Inteligência (30-90 dias)
- [ ] Integrar LSTM Forecaster no pipeline de sinais
- [ ] Implementar GARCH vol forecasting contínuo
- [ ] Feature Store automatizado (update diário)
- [ ] 3+ estratégias backtestadas (Sharpe > 1)
- [ ] Walk-forward validation framework
- [ ] Backtest engine com VectorBT
- [ ] Dashboard Datadog customizado
- [ ] Importar repos existentes como subtrees
- [ ] Notebooks Jupyter para pesquisa
- [ ] Simulação paper-trading (30 dias)

### Fase 3 — Produção (90-180 dias)
- [ ] Aprovação governance: risk committee review
- [ ] Live trading pilot (capital reduzido)
- [ ] Execution engine (Rust, latência < 10ms)
- [ ] Integration com corretora (API B3)
- [ ] Multi-asset: ações, futuros, opções, crypto
- [ ] Auto-rebalancing diário
- [ ] Circuit breaker automático (drawdown > 15%)
- [ ] Relatórios regulatórios (CVM compliance)
- [ ] Scaling: multi-cluster AKS
- [ ] DR: geo-replicação Azure

---

## 9. Matriz de Riscos

| # | Risco | Probabilidade | Impacto | Mitigação |
|---|-------|--------------|---------|-----------|
| R1 | Overfitting LSTM | Alta | Alto | Walk-forward validation, ensemble, regularização |
| R2 | Latência webhook > SLA | Média | Alto | Auto-scaling AKS, cache Redis, rate limiting |
| R3 | Dados corrompidos (yfinance) | Média | Alto | Validação na ingestão, cache parquet, fallback BCB |
| R4 | Drawdown > 15% | Baixa | Crítico | Circuit breaker automático, VaR limits, alertas P1 |
| R5 | Vazamento de credenciais | Baixa | Crítico | Key Vault, SecretStr, audit logs, rotation 90d |
| R6 | Falha MongoDB Atlas | Baixa | Alto | Replica set, backup diário, fallback local |
| R7 | Regime change (mercado) | Alta | Alto | Multi-strategy, GARCH adaptativo, rebalancing |
| R8 | Dependency on TradingView | Média | Médio | Sinais internos (Signal Engine), multi-fonte |
| R9 | CI/CD pipeline quebrado | Média | Médio | Rollback automático, smoke tests, staging |
| R10 | Turnover equipe LMF | Alta | Médio | Documentação detalhada, code reviews, pair programming |

### Plano de Contingência (Circuit Breaker)
```
IF drawdown > -10%:
    → Reduzir alocação 50%
    → Alerta Slack/PagerDuty
    
IF drawdown > -15%:
    → PAUSAR TODAS as operações
    → Notificar Risk Committee
    → Requer aprovação manual para retomar
    
IF VaR diário > -5%:
    → Bloquear novos sinais de COMPRA
    → Aumentar lookback GARCH para 120d
```

---

## 10. Convenções de Desenvolvimento

### Git Workflow
- **main**: Protegida, deploy automático para prod
- **develop**: Integração contínua
- **feature/xxx**: Branches de feature (PR obrigatório)
- **hotfix/xxx**: Correções urgentes → main + develop

### Code Review Checklist
- [ ] Testes unitários passando
- [ ] Cobertura >= 80%
- [ ] ruff + mypy sem erros
- [ ] Documentação atualizada
- [ ] Smoke test de backtest OK
- [ ] Risk metrics dentro dos limites
- [ ] Secrets não hardcoded

### Commits
```
feat(risk): adicionar CVaR condicional ao motor de risco
fix(webhook): corrigir validação IP TradingView
docs(arch): atualizar diagrama de fluxo de sinais
test(signals): adicionar testes para RSI edge cases
```
