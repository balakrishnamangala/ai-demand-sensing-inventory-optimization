"""XGBoost-based demand forecasting model with feature importance."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from src.utils import get_logger, calculate_mape

logger = get_logger("xgboost_model")


class XGBoostForecaster:
    """Gradient boosted tree forecaster for SKU-level demand.

    Handles tabular features from the feature engineering pipeline,
    supports time-series cross-validation, and exposes feature
    importance for interpretability.
    """

    def __init__(self, config: dict):
        self.config = config
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: List[str],
            target_col: str = "quantity_sold",
            group_col: str = "sku") -> "XGBoostForecaster":
        """Train XGBoost models per SKU group.

        Args:
            df: Training dataframe with features and target.
            feature_cols: List of feature column names.
            target_col: Target variable column.
            group_col: Column to group SKUs by.
        """
        self.feature_names = feature_cols
        groups = df[group_col].unique()
        logger.info(f"Training XGBoost models for {len(groups)} groups")

        params = {
            "n_estimators": self.config.get("n_estimators", 500),
            "max_depth": self.config.get("max_depth", 6),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_weight": self.config.get("min_child_weight", 5),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        for group_id in groups:
            group_data = df[df[group_col] == group_id].dropna(subset=feature_cols)
            if len(group_data) < 52:
                logger.warning(f"Skipping {group_id}: insufficient data ({len(group_data)} rows)")
                continue

            X = group_data[feature_cols].values
            y = group_data[target_col].values

            split_idx = int(len(X) * 0.85)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models[group_id] = model

        self.fitted = True
        logger.info(f"Trained {len(self.models)} XGBoost models")
        return self

    def predict(self, df: pd.DataFrame, group_col: str = "sku") -> pd.DataFrame:
        """Generate predictions for each SKU group."""
        if not self.fitted:
            raise RuntimeError("Models must be fitted before prediction")

        predictions = []
        for group_id, model in self.models.items():
            group_data = df[df[group_col] == group_id].copy()
            if len(group_data) == 0:
                continue

            X = group_data[self.feature_names].fillna(0).values
            preds = model.predict(X)
            group_data["forecast"] = np.clip(preds, 0, None)
            predictions.append(group_data)

        return pd.concat(predictions, ignore_index=True)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Aggregate feature importance across all SKU models.

        Returns:
            DataFrame with feature names and average importance scores.
        """
        if not self.models:
            return pd.DataFrame()

        importance_dfs = []
        for group_id, model in self.models.items():
            scores = model.feature_importances_
            imp_df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": scores,
                "group_id": group_id,
            })
            importance_dfs.append(imp_df)

        combined = pd.concat(importance_dfs)
        avg_importance = (
            combined.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return avg_importance

    def cross_validate(self, df: pd.DataFrame, feature_cols: List[str],
                       target_col: str = "quantity_sold",
                       n_splits: int = 5) -> Dict[str, float]:
        """Run time series cross-validation and return average metrics."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X = df[feature_cols].fillna(0).values
        y = df[target_col].values

        mape_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            params = {k: v for k, v in self.config.items()
                      if k not in ("early_stopping_rounds",)}
            params["random_state"] = 42
            params["n_jobs"] = -1

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            preds = np.clip(model.predict(X_test), 0, None)
            mape_scores.append(calculate_mape(y_test, preds))

        return {
            "cv_mape_mean": float(np.mean(mape_scores)),
            "cv_mape_std": float(np.std(mape_scores)),
            "n_splits": n_splits,
        }
