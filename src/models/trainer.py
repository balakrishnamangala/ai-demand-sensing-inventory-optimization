"""Model training orchestrator for demand forecasting pipeline."""
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.ensemble import EnsembleForecaster
from src.utils import get_logger, generate_accuracy_report

logger = get_logger("trainer")


class ModelTrainer:
    """Orchestrates training, evaluation and selection of forecast models."""

    def __init__(self, config: dict):
        self.config = config
        self.prophet = ProphetForecaster(config.get("prophet", {}))
        self.xgboost = XGBoostForecaster(config.get("xgboost", {}))
        self.ensemble = EnsembleForecaster(config.get("ensemble", {
            "prophet_weight": 0.4, "xgboost_weight": 0.6,
            "dynamic_weighting": True,
        }))
        self.results: Dict[str, pd.DataFrame] = {}
        self.metrics: Dict[str, dict] = {}

    def train(self, train_data: pd.DataFrame,
              features: pd.DataFrame,
              group_col: str = "group_id",
              target_col: str = "demand") -> Dict[str, pd.DataFrame]:
        """Train all models and produce ensemble forecast.

        Args:
            train_data: Training data with ds, group_id, y columns.
            features: Engineered features for XGBoost.
            group_col: Column identifying SKU groups.
            target_col: Target variable column name.

        Returns:
            Dictionary of model forecasts keyed by model name.
        """
        logger.info("Starting model training pipeline")
        start = time.time()

        # Prophet training
        logger.info("Training Prophet models...")
        prophet_fc = self.prophet.fit_predict(train_data, group_col=group_col)
        self.results["prophet"] = prophet_fc

        # XGBoost training
        logger.info("Training XGBoost models...")
        xgb_fc = self.xgboost.fit_predict(features, group_col=group_col,
                                           target_col=target_col)
        self.results["xgboost"] = xgb_fc

        # Ensemble
        logger.info("Generating ensemble forecast...")
        actuals = train_data.rename(columns={target_col: "actual"})
        ensemble_fc = self.ensemble.combine(
            prophet_fc, xgb_fc, actuals=actuals, group_col=group_col)
        self.results["ensemble"] = ensemble_fc

        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed:.1f}s")
        return self.results

    def evaluate(self, actuals: pd.DataFrame,
                 group_col: str = "group_id") -> Dict[str, dict]:
        """Evaluate all models against actuals."""
        for name, forecast in self.results.items():
            fc_col = "ensemble_forecast" if name == "ensemble" else "forecast"
            merged = forecast.merge(
                actuals[["ds", group_col, "actual"]],
                on=["ds", group_col], how="inner")
            if len(merged) == 0:
                continue
            report = generate_accuracy_report(
                merged["actual"].values, merged[fc_col].values)
            self.metrics[name] = report
            logger.info(f"{name}: MAPE={report['mape']:.2f}%, "
                        f"RMSE={report['rmse']:.2f}")
        return self.metrics

    def select_best(self) -> str:
        """Select model with lowest MAPE."""
        if not self.metrics:
            return "ensemble"
        best = min(self.metrics, key=lambda k: self.metrics[k]["mape"])
        logger.info(f"Best model: {best} (MAPE={self.metrics[best]['mape']:.2f}%)")
        return best

    def save_forecasts(self, output_dir: str, best_only: bool = False) -> None:
        """Save forecast results to parquet files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        models = {self.select_best(): self.results[self.select_best()]} \
            if best_only else self.results
        for name, df in models.items():
            path = out / f"{name}_forecast.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Saved {name} forecast → {path}")
