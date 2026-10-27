"""Tests for forecast and inventory metrics."""
import numpy as np
from src.utils.metrics import calculate_mape, calculate_rmse, calculate_bias


def test_mape_perfect():
    actual = np.array([100, 200, 300])
    assert calculate_mape(actual, actual) == 0.0


def test_mape_known():
    actual = np.array([100.0, 200.0])
    pred = np.array([110.0, 180.0])
    mape = calculate_mape(actual, pred)
    assert abs(mape - 10.0) < 0.01


def test_rmse_perfect():
    vals = np.array([1, 2, 3])
    assert calculate_rmse(vals, vals) == 0.0


def test_bias_over():
    actual = np.array([100.0])
    pred = np.array([120.0])
    assert calculate_bias(actual, pred) > 0


def test_bias_under():
    actual = np.array([100.0])
    pred = np.array([80.0])
    assert calculate_bias(actual, pred) < 0
