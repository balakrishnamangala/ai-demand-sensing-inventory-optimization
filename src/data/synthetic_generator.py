"""Synthetic demand data generator for testing and demos."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.utils import get_logger

logger = get_logger("synthetic")


def generate_synthetic_data(
    n_skus: int = 50,
    n_weeks: int = 104,
    start_date: str = "2022-01-03",
    seed: int = 42,
) -> dict:
    """Generate realistic synthetic supply chain data.

    Creates correlated demand with seasonality, trend, promotions,
    and weather effects across multiple SKUs and locations.

    Returns:
        Dictionary with DataFrames: sales, promotions, weather,
        lead_times, unit_costs, locations.
    """
    rng = np.random.RandomState(seed)
    start = pd.Timestamp(start_date)
    dates = [start + timedelta(weeks=w) for w in range(n_weeks)]

    skus = [f"SKU-{str(i).zfill(4)}" for i in range(1, n_skus + 1)]

    # Base demand profiles
    base_demands = rng.lognormal(mean=5.0, sigma=0.8, size=n_skus)

    records = []
    for i, sku in enumerate(skus):
        base = base_demands[i]
        trend = rng.uniform(-0.005, 0.01)
        seasonality_amp = rng.uniform(0.1, 0.4)
        noise_level = rng.uniform(0.05, 0.25)

        for w, date in enumerate(dates):
            week_of_year = date.isocalendar()[1]
            seasonal = 1 + seasonality_amp * np.sin(
                2 * np.pi * week_of_year / 52)
            trend_factor = 1 + trend * w
            noise = 1 + rng.normal(0, noise_level)
            promo_lift = 1.0
            if rng.random() < 0.08:
                promo_lift = rng.uniform(1.15, 1.50)

            demand = max(0, base * seasonal * trend_factor * noise * promo_lift)
            records.append({
                "ds": date, "group_id": sku,
                "demand": round(demand, 1),
                "has_promo": promo_lift > 1.0,
                "promo_discount": round((promo_lift - 1) * 0.5, 2) if promo_lift > 1 else 0,
            })

    sales = pd.DataFrame(records)

    # Promotions calendar
    promos = sales[sales["has_promo"]][["ds", "group_id", "promo_discount"]].copy()
    promos["promo_type"] = rng.choice(
        ["price_cut", "bogo", "bundle", "seasonal"], size=len(promos))

    # Weather data
    weather_records = []
    for date in dates:
        week_of_year = date.isocalendar()[1]
        temp = 15 + 15 * np.sin(2 * np.pi * (week_of_year - 10) / 52) + \
               rng.normal(0, 3)
        precip = max(0, rng.exponential(5) + 3 * np.sin(
            2 * np.pi * week_of_year / 52))
        weather_records.append({
            "ds": date, "temperature_c": round(temp, 1),
            "precipitation_mm": round(precip, 1),
        })
    weather = pd.DataFrame(weather_records)

    # Lead times
    lead_times = pd.DataFrame({
        "group_id": skus,
        "avg_lead_time_days": rng.uniform(3, 21, n_skus).round(1),
        "lead_time_std_days": rng.uniform(0.5, 4, n_skus).round(1),
    })

    # Unit costs
    unit_costs = pd.DataFrame({
        "group_id": skus,
        "unit_cost": rng.lognormal(2.5, 0.7, n_skus).round(2),
    })

    # Location tiers
    locations = pd.DataFrame([
        {"location_id": "DC-CENTRAL", "tier": "dc"},
        {"location_id": "RDC-EAST", "tier": "regional"},
        {"location_id": "RDC-WEST", "tier": "regional"},
        {"location_id": "RDC-SOUTH", "tier": "regional"},
    ] + [{"location_id": f"STORE-{str(i).zfill(3)}", "tier": "store"}
         for i in range(1, 16)])

    logger.info(f"Generated synthetic data: {n_skus} SKUs × {n_weeks} weeks")
    return {
        "sales": sales, "promotions": promos, "weather": weather,
        "lead_times": lead_times, "unit_costs": unit_costs,
        "locations": locations,
    }


if __name__ == "__main__":
    data = generate_synthetic_data()
    for name, df in data.items():
        path = f"data/raw/{name}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {name}: {df.shape}")
