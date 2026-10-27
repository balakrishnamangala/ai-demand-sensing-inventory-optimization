"""Main pipeline execution script."""
import sys
import time
from pathlib import Path

from src.data.synthetic_generator import generate_synthetic_data
from src.data.pipeline import DataPipeline
from src.data.feature_engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.optimization.safety_stock import SafetyStockCalculator
from src.optimization.reorder_point import ReorderPointCalculator
from src.optimization.segmentation import ABCXYZSegmenter
from src.optimization.working_capital import WorkingCapitalAnalyzer
from src.utils import get_logger

logger = get_logger("pipeline")


def main():
    logger.info("=" * 60)
    logger.info("AI Demand Sensing & Inventory Optimization Pipeline")
    logger.info("=" * 60)
    start = time.time()

    # Step 1: Generate or load data
    logger.info("Step 1: Loading data...")
    data = generate_synthetic_data(n_skus=30, n_weeks=104)
    sales = data["sales"]
    logger.info(f"  Sales: {sales.shape[0]:,} records, "
                f"{sales['group_id'].nunique()} SKUs")

    # Step 2: Feature engineering
    logger.info("Step 2: Engineering features...")
    fe = FeatureEngineer({})
    features = fe.create_features(sales)
    logger.info(f"  Features: {features.shape[1]} columns")

    # Step 3: ABC/XYZ Segmentation
    logger.info("Step 3: SKU segmentation...")
    seg = ABCXYZSegmenter({})
    segments = seg.classify(sales, data["unit_costs"])
    logger.info(f"  Segments: {segments['segment'].value_counts().to_dict()}")

    # Step 4: Safety stock
    logger.info("Step 4: Calculating safety stock...")
    ss_calc = SafetyStockCalculator({
        "target_service_level": 0.98,
        "method": "demand_variability",
    })
    safety_stock = ss_calc.calculate(sales, data["lead_times"])

    # Step 5: Reorder points
    logger.info("Step 5: Computing reorder points...")
    rop_calc = ReorderPointCalculator({})
    inventory = rop_calc.calculate(safety_stock, data["unit_costs"])

    # Step 6: Working capital analysis
    logger.info("Step 6: Working capital analysis...")
    inv_with_abc = inventory.merge(
        segments[["group_id", "abc_class", "xyz_class", "segment"]],
        on="group_id", how="left")
    wc = WorkingCapitalAnalyzer({"target_reduction_pct": 0.15})
    wc_report = wc.analyze(inv_with_abc, sales)

    # Summary
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info(f"  SKUs processed: {len(inventory)}")
    logger.info(f"  Service level: 98%")
    logger.info(f"  Working capital: ${wc_report['current_working_capital']:,.0f}")
    logger.info(f"  Identified savings: ${wc_report['total_identified_savings']:,.0f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
