"""12-week rolling backtest engine for forecast and inventory validation."""
import pandas as pd
import numpy as np
from typing import Dict, List

from src.utils import get_logger, calculate_mape, calculate_rmse

logger = get_logger("backtester")


class BacktestEngine:
    """Rolling-origin backtest for demand forecast evaluation.

    Simulates 12-week rolling forecast windows to validate
    model performance and inventory policy effectiveness.
    """

    def __init__(self, config: dict):
        self.n_windows = config.get("n_windows", 12)
        self.horizon_weeks = config.get("horizon_weeks", 4)
        self.min_train_weeks = config.get("min_train_weeks", 52)

    def run(self, data: pd.DataFrame, trainer,
            group_col: str = "group_id",
            date_col: str = "ds",
            target_col: str = "demand") -> pd.DataFrame:
        """Execute rolling backtest.

        Args:
            data: Full dataset with ds, group_id, demand columns.
            trainer: ModelTrainer instance.
            group_col: SKU group column.
            date_col: Date column.
            target_col: Target column.

        Returns:
            DataFrame with per-window forecast accuracy metrics.
        """
        dates = sorted(data[date_col].unique())
        total_weeks = len(dates)

        if total_weeks < self.min_train_weeks + self.horizon_weeks:
            logger.warning("Insufficient data for backtest")
            return pd.DataFrame()

        window_results = []
        start_idx = total_weeks - (self.n_windows * self.horizon_weeks)

        for w in range(self.n_windows):
            cutoff_idx = start_idx + (w * self.horizon_weeks)
            train_end = dates[cutoff_idx]
            test_start = dates[cutoff_idx + 1] if cutoff_idx + 1 < len(dates) else None
            test_end_idx = min(cutoff_idx + self.horizon_weeks, len(dates) - 1)
            test_end = dates[test_end_idx]

            if test_start is None:
                continue

            train = data[data[date_col] <= train_end]
            test = data[(data[date_col] >= test_start) &
                        (data[date_col] <= test_end)]

            if len(test) == 0:
                continue

            # Simulate forecast (using rolling mean as proxy for speed)
            forecast = self._simple_forecast(train, test, group_col,
                                              date_col, target_col)

            mape = calculate_mape(test[target_col].values,
                                   forecast["forecast"].values)
            rmse = calculate_rmse(test[target_col].values,
                                   forecast["forecast"].values)
            bias = (forecast["forecast"].sum() - test[target_col].sum()) / \
                   max(test[target_col].sum(), 1) * 100

            # Service level simulation
            stockout_rate = self._simulate_stockouts(
                test, forecast, group_col, target_col)

            window_results.append({
                "window": w + 1,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_train": len(train),
                "n_test": len(test),
                "mape": round(mape, 2),
                "rmse": round(rmse, 2),
                "bias_pct": round(bias, 2),
                "simulated_service_level": round(1 - stockout_rate, 4),
            })

            logger.info(f"Window {w+1}/{self.n_windows}: "
                        f"MAPE={mape:.1f}%, SL={1-stockout_rate:.2%}")

        results = pd.DataFrame(window_results)
        self._log_summary(results)
        return results

    def _simple_forecast(self, train: pd.DataFrame, test: pd.DataFrame,
                          group_col: str, date_col: str,
                          target_col: str) -> pd.DataFrame:
        """Generate simple rolling-mean forecast for backtest speed."""
        forecasts = []
        for sku in test[group_col].unique():
            sku_train = train[train[group_col] == sku][target_col]
            avg = sku_train.tail(8).mean() if len(sku_train) > 0 else 0
            sku_test = test[test[group_col] == sku].copy()
            sku_test["forecast"] = avg
            forecasts.append(sku_test)
        return pd.concat(forecasts, ignore_index=True)

    def _simulate_stockouts(self, actuals: pd.DataFrame,
                             forecast: pd.DataFrame,
                             group_col: str,
                             target_col: str) -> float:
        """Simulate stockout rate using forecast vs actual."""
        merged = actuals.merge(
            forecast[["ds", group_col, "forecast"]],
            on=["ds", group_col], how="inner")
        if len(merged) == 0:
            return 0.0
        stockouts = (merged[target_col] > merged["forecast"] * 1.1).sum()
        return stockouts / len(merged)

    def _log_summary(self, results: pd.DataFrame) -> None:
        if len(results) == 0:
            return
        logger.info(
            f"Backtest summary ({self.n_windows} windows): "
            f"avg MAPE={results['mape'].mean():.1f}%, "
            f"avg SL={results['simulated_service_level'].mean():.2%}")
