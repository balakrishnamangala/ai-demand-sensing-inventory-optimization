"""Forecast accuracy and inventory performance metrics."""
import numpy as np
import pandas as pd
from typing import Optional


def calculate_mape(actual: np.ndarray, forecast: np.ndarray,
                   exclude_zeros: bool = True) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        actual: Array of actual demand values.
        forecast: Array of forecast values.
        exclude_zeros: Whether to exclude zero actuals.

    Returns:
        MAPE as a percentage (0-100).
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if exclude_zeros:
        mask = actual != 0
        actual = actual[mask]
        forecast = forecast[mask]

    if len(actual) == 0:
        return np.nan

    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def calculate_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return float(np.sqrt(np.mean((np.asarray(actual, dtype=float) -
                                  np.asarray(forecast, dtype=float)) ** 2)))


def calculate_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) -
                                np.asarray(forecast, dtype=float))))


def calculate_bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Calculate forecast bias. Positive means over-forecasting."""
    return float(np.mean(np.asarray(forecast, dtype=float) -
                         np.asarray(actual, dtype=float)))


def calculate_tracking_signal(actual: np.ndarray, forecast: np.ndarray,
                               smoothing: float = 0.1) -> np.ndarray:
    """Calculate cumulative tracking signal for forecast monitoring.

    Values beyond +/- 4 indicate systematic bias requiring
    model recalibration or parameter adjustment.
    """
    errors = np.asarray(forecast, dtype=float) - np.asarray(actual, dtype=float)
    cum_error = np.cumsum(errors)
    mad = np.zeros_like(errors, dtype=float)
    mad[0] = abs(errors[0])
    for i in range(1, len(errors)):
        mad[i] = smoothing * abs(errors[i]) + (1 - smoothing) * mad[i - 1]
    return np.where(mad != 0, cum_error / mad, 0)


def calculate_fill_rate(demand: np.ndarray, fulfilled: np.ndarray) -> float:
    """Calculate order fill rate as fulfilled / total demand."""
    total_demand = np.sum(demand)
    if total_demand == 0:
        return 1.0
    return float(np.sum(fulfilled) / total_demand)


def calculate_inventory_turns(cogs: float, avg_inventory_value: float) -> float:
    """Calculate annualized inventory turnover ratio."""
    if avg_inventory_value == 0:
        return np.inf
    return cogs / avg_inventory_value


def calculate_days_of_supply(on_hand: float, avg_daily_demand: float) -> float:
    """Calculate days of supply from current on-hand inventory."""
    if avg_daily_demand <= 0:
        return np.inf
    return on_hand / avg_daily_demand


def calculate_service_level(periods_in_stock: int, total_periods: int) -> float:
    """Calculate cycle service level (CSL) as percentage of periods with stock."""
    if total_periods == 0:
        return 0.0
    return periods_in_stock / total_periods


def generate_accuracy_report(
    actual: pd.Series,
    forecast: pd.Series,
    baseline: Optional[pd.Series] = None
) -> dict:
    """Generate comprehensive forecast accuracy report.

    Returns:
        Dictionary with MAPE, RMSE, MAE, bias, and improvement
        versus the baseline model if provided.
    """
    a = actual.values
    f = forecast.values

    report = {
        "mape": calculate_mape(a, f),
        "rmse": calculate_rmse(a, f),
        "mae": calculate_mae(a, f),
        "bias": calculate_bias(a, f),
        "n_observations": len(a),
    }

    if baseline is not None:
        b = baseline.values
        baseline_mape = calculate_mape(a, b)
        report["baseline_mape"] = baseline_mape
        if baseline_mape > 0:
            report["mape_improvement_pct"] = round(
                (baseline_mape - report["mape"]) / baseline_mape * 100, 2
            )

    return report
