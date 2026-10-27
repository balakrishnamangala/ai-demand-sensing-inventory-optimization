"""Dashboard data service for Power BI / visualization integration."""
import pandas as pd
from typing import Dict

from src.utils import get_logger

logger = get_logger("dashboard")


class DashboardService:
    """Prepare and export data for Power BI dashboards."""

    def generate_forecast_dashboard(self, forecasts: pd.DataFrame,
                                     actuals: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate forecast accuracy dashboard data."""
        merged = forecasts.merge(actuals, on=["ds", "group_id"],
                                  how="left", suffixes=("_fc", "_act"))
        merged["error"] = merged.get("demand", 0) - merged.get("forecast", 0)
        merged["abs_pct_error"] = (
            abs(merged["error"]) / merged["demand"].clip(lower=1) * 100)

        weekly = merged.groupby("ds").agg(
            avg_forecast=("forecast", "mean"),
            avg_actual=("demand", "mean"),
            mape=("abs_pct_error", "mean"),
        ).reset_index()

        return {"forecast_vs_actual": weekly, "detail": merged}

    def generate_inventory_dashboard(self, inventory: pd.DataFrame) -> pd.DataFrame:
        """Generate inventory KPI dashboard data."""
        summary = inventory.groupby("abc_class").agg(
            sku_count=("group_id", "count"),
            total_safety_stock=("safety_stock_units", "sum"),
            total_investment=("inventory_investment", "sum"),
            avg_rop=("reorder_point", "mean"),
        ).reset_index()
        return summary

    def export_to_csv(self, data: Dict[str, pd.DataFrame],
                       output_dir: str = "dashboards") -> None:
        """Export dashboard datasets to CSV for Power BI import."""
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(exist_ok=True)
        for name, df in data.items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info(f"Exported {name} → {path}")
