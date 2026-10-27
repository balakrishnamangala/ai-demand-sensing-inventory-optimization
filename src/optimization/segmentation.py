"""ABC/XYZ SKU segmentation for inventory classification."""
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("segmentation")


class ABCXYZSegmenter:
    """Classify SKUs by revenue contribution (ABC) and demand variability (XYZ).

    ABC: Pareto analysis on revenue/volume
      A = top 80% cumulative revenue
      B = next 15%
      C = remaining 5%

    XYZ: Coefficient of variation of demand
      X = CV < 0.5 (stable)
      Y = 0.5 <= CV < 1.0 (variable)
      Z = CV >= 1.0 (erratic)
    """

    def __init__(self, config: dict):
        self.abc_thresholds = config.get("abc_thresholds", [0.80, 0.95])
        self.xyz_thresholds = config.get("xyz_thresholds", [0.5, 1.0])

    def classify(self, demand_history: pd.DataFrame,
                 unit_costs: pd.DataFrame = None,
                 group_col: str = "group_id",
                 demand_col: str = "demand") -> pd.DataFrame:
        """Run ABC/XYZ segmentation.

        Args:
            demand_history: Historical demand with ds, group_id, demand.
            unit_costs: Optional unit costs for revenue-based ABC.
            group_col: SKU group column.
            demand_col: Demand column name.

        Returns:
            DataFrame with group_id, abc_class, xyz_class, segment.
        """
        agg = demand_history.groupby(group_col).agg(
            total_demand=(demand_col, "sum"),
            avg_demand=(demand_col, "mean"),
            std_demand=(demand_col, "std"),
            periods=(demand_col, "count"),
        ).reset_index()

        agg["std_demand"] = agg["std_demand"].fillna(0)

        # ABC classification
        if unit_costs is not None:
            agg = agg.merge(unit_costs[[group_col, "unit_cost"]],
                            on=group_col, how="left")
            agg["unit_cost"] = agg["unit_cost"].fillna(
                agg["unit_cost"].median())
            agg["revenue"] = agg["total_demand"] * agg["unit_cost"]
        else:
            agg["revenue"] = agg["total_demand"]

        agg = agg.sort_values("revenue", ascending=False)
        agg["cumulative_pct"] = (
            agg["revenue"].cumsum() / agg["revenue"].sum())

        agg["abc_class"] = "C"
        agg.loc[agg["cumulative_pct"] <= self.abc_thresholds[0], "abc_class"] = "A"
        agg.loc[
            (agg["cumulative_pct"] > self.abc_thresholds[0]) &
            (agg["cumulative_pct"] <= self.abc_thresholds[1]),
            "abc_class"] = "B"

        # XYZ classification
        agg["cv"] = np.where(
            agg["avg_demand"] > 0,
            agg["std_demand"] / agg["avg_demand"],
            999.0
        )
        agg["xyz_class"] = "Z"
        agg.loc[agg["cv"] < self.xyz_thresholds[0], "xyz_class"] = "X"
        agg.loc[
            (agg["cv"] >= self.xyz_thresholds[0]) &
            (agg["cv"] < self.xyz_thresholds[1]),
            "xyz_class"] = "Y"

        # Combined segment
        agg["segment"] = agg["abc_class"] + agg["xyz_class"]

        # Inventory policy recommendation
        agg["policy"] = agg["segment"].map(self._policy_map())

        counts = agg["segment"].value_counts().to_dict()
        logger.info(f"Segmentation complete: {counts}")
        return agg

    @staticmethod
    def _policy_map() -> dict:
        """Map segments to inventory policy recommendations."""
        return {
            "AX": "continuous_review_rop",
            "AY": "continuous_review_dynamic_ss",
            "AZ": "make_to_order",
            "BX": "periodic_review_fixed",
            "BY": "periodic_review_variable_ss",
            "BZ": "periodic_review_high_ss",
            "CX": "min_max_auto",
            "CY": "min_max_auto",
            "CZ": "eliminate_or_spot_buy",
        }
