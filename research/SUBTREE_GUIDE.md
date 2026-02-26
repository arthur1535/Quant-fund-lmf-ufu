# =====================================================================
# QuantNucleo — Research: Subtree Integration Guide
# =====================================================================

# Este diretório integra os repositórios de pesquisa existentes como
# subtrees do Git, permitindo reutilização direta do código.

## Comandos para importar repositórios existentes:

```bash
# 1. Monte Carlo & análise energy (eua/)
git subtree add --prefix=research/energy \
    https://github.com/arthur1535/Quant-fund-lmf.uvu.git \
    main --squash

# 2. LSTM Previsão de Carteira
git subtree add --prefix=research/forecasting/portfolio \
    https://github.com/arthur1535/Quant-fund-lmf.uvu.git \
    main --squash

# 3. Para atualizar um subtree:
git subtree pull --prefix=research/energy \
    https://github.com/arthur1535/Quant-fund-lmf.uvu.git \
    main --squash
```

## Estrutura esperada:

```
research/
├── __init__.py
├── forecasting/
│   ├── __init__.py
│   ├── lstm_forecaster.py       ← Wrapper integrado ao pipeline
│   └── portfolio/                ← subtree: LSTM-Previsão-de-Carteira
│       └── portfolio_lstm_garch.py
├── energy/                       ← subtree: análise energia
│   ├── analiseempresasamericanas.py
│   └── analise_slb_detalhada.py
└── notebooks/                    ← Jupyter notebooks de pesquisa
    └── README.md
```
