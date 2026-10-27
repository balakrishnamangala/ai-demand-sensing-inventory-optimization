"""Shared test fixtures."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys

# Mock heavy dependencies that require compilation or system libraries
for mod in [
    'prophet', 'prophet.forecaster', 'prophet.serialize',
    'xgboost', 'xgboost.core', 'xgboost.sklearn',
    'pyodbc',
    'azure', 'azure.storage', 'azure.storage.blob', 'azure.identity',
    'loguru', 'loguru._logger',
    'cmdstanpy',
]:
    sys.modules.setdefault(mod, MagicMock())

# Ensure loguru mock has a usable logger attribute
sys.modules['loguru'].logger = MagicMock()


@pytest.fixture
def sample_demand():
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=52, freq="W")
    records = []
    for sku in ["SKU-0001", "SKU-0002", "SKU-0003"]:
        base = rng.uniform(80, 200)
        for d in dates:
            demand = max(0, base + rng.normal(0, base * 0.15))
            records.append({"ds": d, "group_id": sku, "demand": round(demand, 1)})
    return pd.DataFrame(records)


@pytest.fixture
def sample_lead_times():
    return pd.DataFrame({
        "group_id": ["SKU-0001", "SKU-0002", "SKU-0003"],
        "avg_lead_time_days": [7.0, 14.0, 5.0],
        "lead_time_std_days": [1.5, 3.0, 1.0],
    })


@pytest.fixture
def sample_unit_costs():
    return pd.DataFrame({
        "group_id": ["SKU-0001", "SKU-0002", "SKU-0003"],
        "unit_cost": [25.0, 150.0, 8.50],
    })
