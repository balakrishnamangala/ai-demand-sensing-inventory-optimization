"""Prophet-based demand forecasting model."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from prophet import Prophet

from src.utils import get_logger, calculate_mape

logger = get_logger("prophet_model")


class ProphetForecaster:
    """Demand forecaster using Facebook Prophet.

    Handles trend decomposition, multiple seasonality,
    holiday effects, and external regressors.
    """

    def __init__(self, config: dict):
        self.config = config
        self.models: Dict[str, Prophet] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame, date_col: str = "period",
            target_col: str = "quantity_sold", group_col: str = "sku",
            regressors: Optional[List[str]] = None) -> "ProphetForecaster":
        """Fit Prophet models per SKU group."""
        groups = df[group_col].unique()
        logger.info(f"Training Prophet models for {len(groups)} groups")

        for group_id in groups:
            group_data = df[df[group_col] == group_id].copy()
            if len(group_data) < 26:
                logger.warning(f"Skipping {group_id}: insufficient data ({len(group_data)} rows)")
                continue

            prophet_df = pd.DataFrame({
                "ds": pd.to_datetime(group_data[date_col]),
                "y": group_data[target_col].values,
            })

            seasonality = self.config.get("seasonality", {})
            model = Prophet(
                yearly_seasonality=seasonality.get("yearly", True),
                weekly_seasonality=seasonality.get("weekly", True),
                daily_seasonality=seasonality.get("daily", False),
                changepoint_prior_scale=self.config.get("changepoint_prior_scale", 0.05),
                seasonality_prior_scale=self.config.get("seasonality_prior_scale", 10),
                interval_width=self.config.get("interval_width", 0.95),
            )

            country = self.config.get("holidays_country")
            if country:
                model.add_country_holidays(country_name=country)

            if regressors:
                for reg in regressors:
                    if reg in group_data.columns:
                        prophet_df[reg] = group_data[reg].values
                        model.add_regressor(reg)

            model.fit(prophet_df)
            self.models[group_id] = model

        self.fitted = True
        logger.info(f"Trained {len(self.models)} Prophet models")
        return self

    def predict(self, horizon_periods: int,
                regressors_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate forecasts for all fitted models."""
        if not self.fitted:
            raise RuntimeError("Models must be fitted before prediction")

        all_forecasts = []
        for group_id, model in self.models.items():
            future = model.make_future_dataframe(periods=horizon_periods, freq="W")

            if regressors_df is not None:
                for col in regressors_df.columns:
                    if col != "ds":
                        future = future.merge(regressors_df[["ds", col]], on="ds", how="left")
                        future[col] = future[col].ffill().fillna(0)

            forecast = model.predict(future)
            forecast["group_id"] = group_id
            forecast["forecast"] = forecast["yhat"].clip(lower=0)
            forecast["lower"] = forecast["yhat_lower"].clip(lower=0)
            forecast["upper"] = forecast["yhat_upper"].clip(lower=0)

            all_forecasts.append(
                forecast[["ds", "group_id", "forecast", "lower", "upper", "trend", "yearly"]]
            )

        return pd.concat(all_forecasts, ignore_index=True)

    def evaluate(self, actual: pd.DataFrame, forecast: pd.DataFrame,
                 date_col: str = "ds", target_col: str = "y") -> Dict[str, float]:
        """Evaluate forecast accuracy against actuals."""
        merged = actual.merge(forecast[["ds", "group_id", "forecast"]],
                              left_on=[date_col, "group_id"],
                              right_on=["ds", "group_id"])
        return {
            "mape": calculate_mape(merged[target_col].values, merged["forecast"].values),
            "n_groups": merged["group_id"].nunique(),
            "n_observations": len(merged),
        }
