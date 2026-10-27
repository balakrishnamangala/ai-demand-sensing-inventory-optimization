"""Ensemble forecaster combining Prophet and XGBoost predictions."""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.utils import get_logger, calculate_mape

logger = get_logger("ensemble")


class EnsembleForecaster:
    """Weighted ensemble combining Prophet and XGBoost forecasts.

    Supports static weight assignment and dynamic weighting
    based on recent cross-validation performance per SKU.
    """

    def __init__(self, config: dict):
        self.config = config
        self.prophet_weight = config.get("prophet_weight", 0.4)
        self.xgboost_weight = config.get("xgboost_weight", 0.6)
        self.dynamic_weighting = config.get("dynamic_weighting", True)
        self.sku_weights: Dict[str, Dict[str, float]] = {}

    def combine(self, prophet_forecast: pd.DataFrame,
                xgboost_forecast: pd.DataFrame,
                actuals: Optional[pd.DataFrame] = None,
                group_col: str = "group_id") -> pd.DataFrame:
        """Combine Prophet and XGBoost forecasts.

        Args:
            prophet_forecast: DataFrame with Prophet predictions.
            xgboost_forecast: DataFrame with XGBoost predictions.
            actuals: Optional actuals for dynamic weight calculation.
            group_col: Column identifying SKU groups.

        Returns:
            DataFrame with ensemble forecast and component weights.
        """
        merged = prophet_forecast[["ds", group_col, "forecast"]].rename(
            columns={"forecast": "prophet_forecast"}
        ).merge(
            xgboost_forecast[["ds", group_col, "forecast"]].rename(
                columns={"forecast": "xgboost_forecast"}
            ),
            on=["ds", group_col],
            how="inner",
        )

        if self.dynamic_weighting and actuals is not None:
            self._calculate_dynamic_weights(merged, actuals, group_col)

        merged["ensemble_forecast"] = merged.apply(
            lambda row: self._weighted_average(row, group_col), axis=1
        )
        merged["ensemble_forecast"] = merged["ensemble_forecast"].clip(lower=0)

        logger.info(f"Ensemble forecast generated: {len(merged):,} predictions")
        return merged

    def _calculate_dynamic_weights(self, forecasts: pd.DataFrame,
                                    actuals: pd.DataFrame,
                                    group_col: str) -> None:
        """Calculate per-SKU weights based on recent forecast accuracy."""
        for group_id in forecasts[group_col].unique():
            group_fc = forecasts[forecasts[group_col] == group_id]
            group_act = actuals[actuals[group_col] == group_id]

            if len(group_act) < 4:
                continue

            merged = group_fc.merge(group_act[["ds", group_col, "actual"]],
                                     on=["ds", group_col], how="inner")
            if len(merged) == 0:
                continue

            prophet_mape = calculate_mape(
                merged["actual"].values, merged["prophet_forecast"].values)
            xgb_mape = calculate_mape(
                merged["actual"].values, merged["xgboost_forecast"].values)

            if prophet_mape + xgb_mape == 0:
                continue

            inv_prophet = 1.0 / max(prophet_mape, 0.01)
            inv_xgb = 1.0 / max(xgb_mape, 0.01)
            total = inv_prophet + inv_xgb

            self.sku_weights[group_id] = {
                "prophet": inv_prophet / total,
                "xgboost": inv_xgb / total,
            }

    def _weighted_average(self, row, group_col: str) -> float:
        """Calculate weighted average for a single prediction row."""
        group_id = row[group_col]
        if group_id in self.sku_weights:
            w = self.sku_weights[group_id]
            pw, xw = w["prophet"], w["xgboost"]
        else:
            pw, xw = self.prophet_weight, self.xgboost_weight

        return pw * row["prophet_forecast"] + xw * row["xgboost_forecast"]

    def get_weights_summary(self) -> pd.DataFrame:
        """Return summary of per-SKU dynamic weights."""
        if not self.sku_weights:
            return pd.DataFrame({
                "method": ["static"],
                "prophet_weight": [self.prophet_weight],
                "xgboost_weight": [self.xgboost_weight],
            })

        records = []
        for sku, weights in self.sku_weights.items():
            records.append({
                "sku": sku,
                "prophet_weight": round(weights["prophet"], 3),
                "xgboost_weight": round(weights["xgboost"], 3),
            })
        return pd.DataFrame(records)
