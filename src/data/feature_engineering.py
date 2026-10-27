"""Feature engineering for demand sensing models."""
import pandas as pd
import numpy as np
from typing import List, Optional

from src.utils import get_logger

logger = get_logger("feature_engineering")


class FeatureEngineer:
    """Generate time series and demand features for forecasting.

    Creates lag features, rolling statistics, calendar encodings,
    trend indicators, and volatility measures.
    """

    def __init__(self, target_col: str = "quantity_sold",
                 date_col: str = "period",
                 group_cols: Optional[List[str]] = None):
        self.target_col = target_col
        self.date_col = date_col
        self.group_cols = group_cols or ["sku", "location"]

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all engineered features."""
        logger.info("Starting feature generation")
        df = df.copy().sort_values(self.group_cols + [self.date_col])

        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_calendar_features(df)
        df = self._add_trend_features(df)
        df = self._add_demand_volatility(df)

        feature_count = len(self.get_feature_names(df))
        logger.info(f"Generated {feature_count} features")
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged demand features at 1w, 2w, 4w, 8w, 13w, 26w, 52w."""
        grouped = df.groupby(self.group_cols)
        for lag in [1, 2, 4, 8, 13, 26, 52]:
            df[f"demand_lag_{lag}w"] = grouped[self.target_col].shift(lag)

        df["demand_yoy_diff"] = (
            df[self.target_col] - df.groupby(self.group_cols)[self.target_col].shift(52)
        )
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics (mean, std, min, max)."""
        grouped = df.groupby(self.group_cols)[self.target_col]
        for w in [4, 8, 13, 26]:
            df[f"roll_mean_{w}w"] = grouped.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            df[f"roll_std_{w}w"] = grouped.transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).std())
            df[f"roll_min_{w}w"] = grouped.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).min())
            df[f"roll_max_{w}w"] = grouped.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).max())

        df["ewm_4w"] = grouped.transform(
            lambda x: x.shift(1).ewm(span=4, min_periods=1).mean())
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar and Fourier seasonality features."""
        dt = pd.to_datetime(df[self.date_col])
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["year"] = dt.dt.year

        df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

        df["is_q4"] = (df["quarter"] == 4).astype(int)
        df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add linear trend and momentum indicators."""
        grouped = df.groupby(self.group_cols)
        df["pct_change_1w"] = grouped[self.target_col].pct_change(1)
        df["pct_change_4w"] = grouped[self.target_col].pct_change(4)
        df["momentum_4w"] = grouped[self.target_col].transform(
            lambda x: x.rolling(4, min_periods=2).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) >= 2 else 0, raw=True))
        df["time_index"] = grouped.cumcount()
        return df

    def _add_demand_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add coefficient of variation and intermittency measures."""
        grouped = df.groupby(self.group_cols)[self.target_col]
        mean_13 = grouped.transform(lambda x: x.shift(1).rolling(13, min_periods=4).mean())
        std_13 = grouped.transform(lambda x: x.shift(1).rolling(13, min_periods=4).std())
        df["cv_13w"] = np.where(mean_13 > 0, std_13 / mean_13, 0)
        df["intermittency"] = grouped.transform(
            lambda x: x.shift(1).rolling(13, min_periods=4).apply(
                lambda v: (v == 0).sum() / len(v), raw=True))
        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return list of generated feature column names."""
        exclude = set(self.group_cols + [self.date_col, self.target_col])
        return [c for c in df.columns if c not in exclude]
