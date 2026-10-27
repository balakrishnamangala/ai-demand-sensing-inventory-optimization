# API Reference

## Base URL
`http://localhost:8000`

## Endpoints

### Health Check
`GET /health`

Returns service status and model readiness.

### Generate Forecast
`POST /api/v1/forecast`

**Request:**
```json
{
  "sku_ids": ["SKU-0001", "SKU-0002"],
  "horizon_weeks": 12,
  "include_components": true
}
```

**Response:** Ensemble forecasts with optional Prophet/XGBoost components.

### Optimize Inventory
`POST /api/v1/inventory/optimize`

**Request:**
```json
{
  "sku_ids": ["SKU-0001"],
  "service_level": 0.98
}
```

**Response:** Safety stock, ROP, EOQ, ABC/XYZ class and policy.

### Train Models
`POST /api/v1/train`

Triggers retraining with latest data. Returns training timestamp.

### Backtest Results
`GET /api/v1/backtest`

Returns 12-week rolling backtest accuracy metrics.
