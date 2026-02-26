# QuantNucleo — Configuração de Webhooks TradingView
# =====================================================
# Como configurar alertas no TradingView para integrar com o núcleo quant.

## Pré-requisitos

1. Conta TradingView (Pro+ recomendado para mais alertas)
2. Endpoint do webhook deployed (Azure Functions ou AKS)
3. Token de autenticação configurado

## Configuração do Webhook

### URL do Endpoint
```
# Desenvolvimento
http://localhost:8001/api/v1/webhooks/tradingview

# Staging
https://quantnucleo-staging.azurewebsites.net/api/v1/webhooks/tradingview

# Produção
https://quantnucleo.azurewebsites.net/api/v1/webhooks/tradingview
```

### Headers Obrigatórios
```
X-Webhook-Token: <seu-token-secreto>
Content-Type: application/json
```

### Formato do Payload JSON
```json
{
  "ticker": "{{ticker}}",
  "action": "buy",
  "price": {{close}},
  "interval": "{{interval}}",
  "exchange": "{{exchange}}",
  "strategy_name": "nome_da_estrategia",
  "indicator_value": null,
  "message": "Descrição do sinal",
  "idempotency_key": "{{ticker}}_{{timenow}}_buy"
}
```

### Campos Possíveis para `action`
- `"buy"` → Sinal de compra
- `"sell"` → Sinal de venda
- `"alert"` → Alerta informativo (sem ação)

### Variáveis do TradingView Disponíveis
| Variável | Descrição |
|----------|-----------|
| `{{ticker}}` | Símbolo do ativo |
| `{{close}}` | Preço de fechamento |
| `{{open}}` | Preço de abertura |
| `{{high}}` | Máxima |
| `{{low}}` | Mínima |
| `{{volume}}` | Volume |
| `{{time}}` | Timestamp |
| `{{timenow}}` | Timestamp atual |
| `{{interval}}` | Timeframe |
| `{{exchange}}` | Exchange |

## Testando o Webhook

```bash
# Teste local
curl -X POST http://localhost:8001/api/v1/webhooks/tradingview \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Token: seu-token" \
  -d '{
    "ticker": "PETR4.SA",
    "action": "buy",
    "price": 38.50,
    "interval": "1D",
    "exchange": "BMFBOVESPA",
    "strategy_name": "test",
    "message": "Teste webhook"
  }'
```

## Segurança

- **Token**: Gere um token forte com `openssl rand -hex 32`
- **IPs TradingView**: Permitir apenas: `52.89.214.238`, `34.212.75.30`, `54.218.53.128`, `52.32.178.7`
- **HTTPS**: Sempre use HTTPS em staging/produção
- **Rate Limit**: Configure rate limiting no API Gateway / Azure Front Door
