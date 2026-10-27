"""Working capital optimization and reduction analysis."""
import pandas as pd
import numpy as np

from src.utils import get_logger

logger = get_logger("working_capital")


class WorkingCapitalAnalyzer:
    """Analyze and optimize working capital tied in inventory.

    Identifies reduction opportunities through:
    - Safety stock right-sizing
    - Lead time reduction impact
    - Service level trade-off analysis
    - Dead/slow-moving stock identification
    """

    def __init__(self, config: dict):
        self.target_reduction_pct = config.get("target_reduction_pct", 0.15)
        self.slow_moving_threshold_weeks = config.get(
            "slow_moving_threshold_weeks", 12)

    def analyze(self, inventory_params: pd.DataFrame,
                demand_history: pd.DataFrame,
                group_col: str = "group_id") -> dict:
        """Run working capital analysis.

        Args:
            inventory_params: Current inventory parameters with costs.
            demand_history: Historical demand data.
            group_col: SKU group column.

        Returns:
            Analysis report with reduction opportunities.
        """
        current_investment = inventory_params["inventory_investment"].sum()

        # Identify slow movers
        recent = demand_history.sort_values("ds").groupby(group_col).tail(
            self.slow_moving_threshold_weeks)
        avg_recent = recent.groupby(group_col)["demand"].mean()
        slow_movers = avg_recent[avg_recent < avg_recent.quantile(0.1)].index

        slow_investment = inventory_params[
            inventory_params[group_col].isin(slow_movers)
        ]["inventory_investment"].sum()

        # Lead time reduction impact (10% LT cut → ~5% SS reduction)
        lt_reduction_savings = current_investment * 0.05

        # Service level trade-off (98% → 95% saves ~18% SS)
        sl_tradeoff_savings = current_investment * 0.18

        # ABC-driven reduction (reduce C-item SS by 30%)
        if "abc_class" in inventory_params.columns:
            c_items = inventory_params[
                inventory_params["abc_class"] == "C"]
            c_reduction = c_items["inventory_investment"].sum() * 0.30
        else:
            c_reduction = 0

        target_savings = current_investment * self.target_reduction_pct

        report = {
            "current_working_capital": round(current_investment, 2),
            "target_reduction_pct": self.target_reduction_pct,
            "target_savings": round(target_savings, 2),
            "opportunities": {
                "slow_mover_liquidation": round(slow_investment, 2),
                "lead_time_reduction": round(lt_reduction_savings, 2),
                "service_level_tradeoff": round(sl_tradeoff_savings, 2),
                "c_item_rationalization": round(c_reduction, 2),
            },
            "total_identified_savings": round(
                slow_investment + lt_reduction_savings + c_reduction, 2),
            "slow_mover_skus": list(slow_movers),
            "n_slow_movers": len(slow_movers),
        }

        logger.info(
            f"Working capital: ${current_investment:,.0f}, "
            f"identified savings: ${report['total_identified_savings']:,.0f}")
        return report
