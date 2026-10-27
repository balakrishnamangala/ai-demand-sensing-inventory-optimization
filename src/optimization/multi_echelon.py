"""Multi-echelon inventory optimization across DC, regional, store tiers."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List

from src.utils import get_logger

logger = get_logger("multi_echelon")


class MultiEchelonOptimizer:
    """Optimize inventory across a 3-tier distribution network.

    Tiers:
    - DC (Distribution Center): Central warehouse
    - Regional: Regional distribution hubs
    - Store: Retail store locations

    Applies differentiated service levels and safety stock
    multipliers at each echelon based on demand aggregation.
    """

    def __init__(self, config: dict):
        self.tiers = config.get("tiers", {
            "dc": {"service_level": 0.99, "lead_time_days": 14,
                   "holding_cost_mult": 0.7},
            "regional": {"service_level": 0.98, "lead_time_days": 5,
                         "holding_cost_mult": 0.85},
            "store": {"service_level": 0.95, "lead_time_days": 2,
                      "holding_cost_mult": 1.0},
        })
        self.network: Dict[str, List[str]] = {}

    def define_network(self, network_map: Dict[str, List[str]]) -> None:
        """Define network hierarchy.

        Args:
            network_map: e.g. {"DC-East": ["RDC-NE", "RDC-SE"],
                               "RDC-NE": ["Store-001", "Store-002"]}
        """
        self.network = network_map
        logger.info(f"Network defined: {len(network_map)} nodes")

    def optimize(self, demand_by_location: pd.DataFrame,
                 location_tiers: pd.DataFrame,
                 group_col: str = "group_id") -> pd.DataFrame:
        """Optimize inventory allocation across echelons.

        Args:
            demand_by_location: Demand with location_id, group_id, demand.
            location_tiers: location_id → tier mapping.
            group_col: SKU group column.

        Returns:
            Optimized inventory parameters per location-SKU.
        """
        merged = demand_by_location.merge(
            location_tiers[["location_id", "tier"]], on="location_id")

        results = []
        for (loc, sku), grp in merged.groupby(["location_id", group_col]):
            tier = grp["tier"].iloc[0]
            tier_config = self.tiers.get(tier, self.tiers["store"])

            avg_demand = grp["demand"].mean()
            std_demand = grp["demand"].std() if len(grp) > 1 else avg_demand * 0.3

            sl = tier_config["service_level"]
            lt = tier_config["lead_time_days"]
            z = stats.norm.ppf(sl)

            # Risk pooling benefit at higher echelons
            n_children = len(self.network.get(loc, []))
            pooling_factor = 1.0 / np.sqrt(max(n_children, 1))

            safety_stock = z * std_demand * np.sqrt(lt) * pooling_factor
            reorder_point = avg_demand * lt + safety_stock

            holding_mult = tier_config["holding_cost_mult"]

            results.append({
                "location_id": loc,
                group_col: sku,
                "tier": tier,
                "avg_daily_demand": round(avg_demand, 2),
                "safety_stock": round(max(safety_stock, 0), 0),
                "reorder_point": round(max(reorder_point, 0), 0),
                "service_level": sl,
                "lead_time_days": lt,
                "pooling_factor": round(pooling_factor, 3),
                "holding_cost_multiplier": holding_mult,
            })

        result_df = pd.DataFrame(results)
        self._log_summary(result_df)
        return result_df

    def calculate_working_capital(self, inventory_df: pd.DataFrame,
                                   unit_costs: pd.DataFrame,
                                   group_col: str = "group_id") -> pd.DataFrame:
        """Calculate working capital tied up in inventory by tier."""
        merged = inventory_df.merge(
            unit_costs[[group_col, "unit_cost"]], on=group_col, how="left")
        merged["unit_cost"] = merged["unit_cost"].fillna(10.0)

        merged["safety_stock_value"] = (
            merged["safety_stock"] * merged["unit_cost"] *
            merged["holding_cost_multiplier"])

        summary = merged.groupby("tier").agg(
            total_safety_stock=("safety_stock", "sum"),
            total_investment=("safety_stock_value", "sum"),
            location_count=("location_id", "nunique"),
            sku_count=(group_col, "nunique"),
        ).reset_index()

        logger.info(f"Working capital: ${summary['total_investment'].sum():,.0f}")
        return summary

    def _log_summary(self, df: pd.DataFrame) -> None:
        for tier in df["tier"].unique():
            tier_data = df[df["tier"] == tier]
            logger.info(
                f"  {tier}: {len(tier_data)} location-SKUs, "
                f"avg SS={tier_data['safety_stock'].mean():.0f}")
