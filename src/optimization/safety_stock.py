"""Safety stock calculation engine with multiple methodologies."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

from src.utils import get_logger

logger = get_logger("safety_stock")


class SafetyStockCalculator:
    """Calculate safety stock using statistical and service-level methods.

    Supports:
    - Standard deviation method
    - Service level (z-score) approach
    - Lead time demand variability
    - Forecast error-based safety stock
    """

    def __init__(self, config: dict):
        self.target_service_level = config.get("target_service_level", 0.98)
        self.review_period_days = config.get("review_period_days", 7)
        self.z_score = stats.norm.ppf(self.target_service_level)
        self.method = config.get("method", "forecast_error")

    def calculate(self, demand_history: pd.DataFrame,
                  lead_times: pd.DataFrame,
                  forecast_errors: Optional[pd.DataFrame] = None,
                  group_col: str = "group_id") -> pd.DataFrame:
        """Calculate safety stock for each SKU.

        Args:
            demand_history: Historical demand with ds, group_id, demand.
            lead_times: Lead time data with group_id, avg_lead_time_days,
                        lead_time_std_days.
            forecast_errors: Forecast residuals per SKU.
            group_col: SKU group column.

        Returns:
            DataFrame with group_id, safety_stock_units, method.
        """
        results = []
        for sku in demand_history[group_col].unique():
            sku_demand = demand_history[demand_history[group_col] == sku]["demand"]
            sku_lt = lead_times[lead_times[group_col] == sku]

            if len(sku_lt) == 0:
                avg_lt = 7.0
                std_lt = 1.5
            else:
                avg_lt = sku_lt["avg_lead_time_days"].iloc[0]
                std_lt = sku_lt["lead_time_std_days"].iloc[0]

            avg_demand = sku_demand.mean()
            std_demand = sku_demand.std()

            if self.method == "forecast_error" and forecast_errors is not None:
                ss = self._forecast_error_method(
                    forecast_errors, sku, avg_lt, group_col)
            else:
                ss = self._demand_variability_method(
                    avg_demand, std_demand, avg_lt, std_lt)

            results.append({
                group_col: sku,
                "safety_stock_units": max(round(ss, 0), 0),
                "avg_daily_demand": round(avg_demand, 2),
                "demand_std": round(std_demand, 2),
                "avg_lead_time_days": avg_lt,
                "lead_time_std_days": std_lt,
                "service_level": self.target_service_level,
                "z_score": round(self.z_score, 3),
                "method": self.method,
            })

        df = pd.DataFrame(results)
        logger.info(f"Safety stock calculated for {len(df)} SKUs, "
                    f"SL={self.target_service_level*100:.0f}%")
        return df

    def _demand_variability_method(self, avg_demand: float,
                                    std_demand: float,
                                    avg_lt: float,
                                    std_lt: float) -> float:
        """Combined demand and lead time variability method.

        SS = z * sqrt(LT * σ_d² + d̄² * σ_lt²)
        """
        variance = (avg_lt * std_demand**2) + (avg_demand**2 * std_lt**2)
        return self.z_score * np.sqrt(max(variance, 0))

    def _forecast_error_method(self, forecast_errors: pd.DataFrame,
                                sku: str, avg_lt: float,
                                group_col: str) -> float:
        """Safety stock based on forecast error distribution.

        SS = z * MAD_forecast * sqrt(LT / review_period)
        """
        sku_errors = forecast_errors[
            forecast_errors[group_col] == sku]["error"]
        if len(sku_errors) == 0:
            return 0.0
        mad = np.mean(np.abs(sku_errors))
        lt_factor = np.sqrt(avg_lt / max(self.review_period_days, 1))
        return self.z_score * mad * lt_factor
