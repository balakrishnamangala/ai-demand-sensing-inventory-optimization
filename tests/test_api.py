"""Tests for FastAPI endpoints."""
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_train():
    resp = client.post("/api/v1/train")
    assert resp.status_code == 200
    assert resp.json()["status"] == "training_complete"


def test_forecast_after_train():
    client.post("/api/v1/train")
    resp = client.post("/api/v1/forecast", json={
        "sku_ids": ["SKU-0001"], "horizon_weeks": 4})
    assert resp.status_code == 200
    assert len(resp.json()["forecasts"]) == 4


def test_inventory_optimize():
    resp = client.post("/api/v1/inventory/optimize", json={
        "sku_ids": ["SKU-0001", "SKU-0002"], "service_level": 0.98})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["inventory_params"]) == 2


def test_backtest():
    resp = client.get("/api/v1/backtest")
    assert resp.status_code == 200
    assert "avg_mape" in resp.json()
