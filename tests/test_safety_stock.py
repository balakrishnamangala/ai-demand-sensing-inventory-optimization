"""Tests for safety stock calculations."""
from src.optimization.safety_stock import SafetyStockCalculator


def test_safety_stock_basic(sample_demand, sample_lead_times):
    calc = SafetyStockCalculator({
        "target_service_level": 0.95,
        "method": "demand_variability",
    })
    result = calc.calculate(sample_demand, sample_lead_times)
    assert len(result) == 3
    assert all(result["safety_stock_units"] >= 0)
    assert all(result["service_level"] == 0.95)


def test_higher_sl_more_stock(sample_demand, sample_lead_times):
    calc_95 = SafetyStockCalculator({"target_service_level": 0.95, "method": "demand_variability"})
    calc_99 = SafetyStockCalculator({"target_service_level": 0.99, "method": "demand_variability"})
    ss_95 = calc_95.calculate(sample_demand, sample_lead_times)
    ss_99 = calc_99.calculate(sample_demand, sample_lead_times)
    assert ss_99["safety_stock_units"].sum() > ss_95["safety_stock_units"].sum()
