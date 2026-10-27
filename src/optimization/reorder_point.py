"""Reorder point and economic order quantity calculations."""
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("reorder_point")


class ReorderPointCalculator:
    """Calculate reorder points and EOQ for inventory replenishment.

    ROP = (Average Daily Demand × Lead Time) + Safety Stock
    EOQ = sqrt(2 * D * S / H)
    """

    def __init__(self, config: dict):
        self.ordering_cost = config.get("ordering_cost_per_order", 50.0)
        self.holding_cost_pct = config.get("annual_holding_cost_pct", 0.25)
        self.working_days = config.get("working_days_per_year", 252)

    def calculate(self, safety_stock_df: pd.DataFrame,
                  unit_costs: Optional[pd.DataFrame] = None,
                  group_col: str = "group_id") -> pd.DataFrame:
        """Calculate reorder points and EOQ.

        Args:
            safety_stock_df: Output from SafetyStockCalculator.
            unit_costs: DataFrame with group_id, unit_cost.
            group_col: SKU group column.

        Returns:
            DataFrame with ROP, EOQ and replenishment parameters.
        """
        df = safety_stock_df.copy()

        # Reorder point
        df["reorder_point"] = (
            df["avg_daily_demand"] * df["avg_lead_time_days"]
            + df["safety_stock_units"]
        ).round(0)

        # EOQ calculation
        if unit_costs is not None:
            df = df.merge(unit_costs[[group_col, "unit_cost"]],
                          on=group_col, how="left")
        else:
            df["unit_cost"] = 10.0  # default

        annual_demand = df["avg_daily_demand"] * self.working_days
        holding_cost = df["unit_cost"] * self.holding_cost_pct

        df["eoq"] = np.sqrt(
            (2 * annual_demand * self.ordering_cost) /
            holding_cost.clip(lower=0.01)
        ).round(0)

        # Max inventory level
        df["max_inventory"] = df["reorder_point"] + df["eoq"]

        # Average cycle stock
        df["avg_cycle_stock"] = (df["eoq"] / 2).round(0)

        # Total average inventory
        df["avg_total_inventory"] = (
            df["avg_cycle_stock"] + df["safety_stock_units"])

        # Annual ordering frequency
        df["orders_per_year"] = (annual_demand / df["eoq"].clip(lower=1)).round(1)

        # Inventory investment
        df["inventory_investment"] = (
            df["avg_total_inventory"] * df["unit_cost"]).round(2)

        logger.info(f"Reorder points calculated for {len(df)} SKUs, "
                    f"total investment=${df['inventory_investment'].sum():,.0f}")
        return df


from typing import Optional
