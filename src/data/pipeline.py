"""ETL pipeline for demand sensing data ingestion and processing."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from src.utils import get_logger, load_config

logger = get_logger("data_pipeline")


class DataPipeline:
    """End-to-end data pipeline for demand sensing.

    Handles ingestion, cleaning, transformation, and output
    from sales history, promotions, weather, and macro data.
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config = load_config(config_path)
        self.data_sources = self.config["data_sources"]
        self.processing = self.config["processing"]

    def run(self) -> pd.DataFrame:
        """Execute the full ETL pipeline."""
        logger.info("Starting data pipeline execution")

        sales = self._load_sales_data()
        logger.info(f"Loaded sales data: {len(sales):,} records")

        sales = self._clean_sales_data(sales)
        promotions = self._load_promotions()
        weather = self._load_weather_data()
        macro = self._load_macro_indicators()

        merged = self._merge_datasets(sales, promotions, weather, macro)
        logger.info(f"Merged dataset: {len(merged):,} records, {merged.shape[1]} features")

        aggregated = self._aggregate(merged)
        self._save_processed(aggregated)
        logger.info("Pipeline execution complete")
        return aggregated

    def _load_sales_data(self) -> pd.DataFrame:
        """Load and parse sales history from configured source."""
        source = self.data_sources["sales"]
        path = Path(source["path"])

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        df[source["date_column"]] = pd.to_datetime(df[source["date_column"]])
        lookback = source["lookback_years"]
        cutoff = datetime.now() - timedelta(days=lookback * 365)
        df = df[df[source["date_column"]] >= cutoff]
        return df

    def _clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sales data: nulls, outliers, negative values."""
        initial_count = len(df)
        df = df.dropna(subset=["sku_id", "location_id", "quantity_sold"])
        df = df[df["quantity_sold"] >= 0]

        if self.processing["outlier_method"] == "iqr":
            threshold = self.processing["outlier_threshold"]
            grouped = df.groupby("sku_id")["quantity_sold"]
            q1 = grouped.transform("quantile", 0.25)
            q3 = grouped.transform("quantile", 0.75)
            iqr = q3 - q1
            df = df[(df["quantity_sold"] >= q1 - threshold * iqr) &
                     (df["quantity_sold"] <= q3 + threshold * iqr)]

        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed:,} records during cleaning ({removed/initial_count:.1%})")
        return df

    def _load_promotions(self) -> Optional[pd.DataFrame]:
        """Load promotional calendar data."""
        source = self.data_sources.get("promotions")
        if not source:
            return None
        path = Path(source["path"])
        if not path.exists():
            logger.warning(f"Promotions file not found: {path}")
            return None
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df[source["date_column"]] = pd.to_datetime(df[source["date_column"]])
        return df

    def _load_weather_data(self) -> Optional[pd.DataFrame]:
        """Load daily weather station data."""
        source = self.data_sources.get("weather")
        if not source:
            return None
        path = Path(source["path"])
        if not path.exists():
            return None
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df[source["date_column"]] = pd.to_datetime(df[source["date_column"]])
        return df

    def _load_macro_indicators(self) -> Optional[pd.DataFrame]:
        """Load macroeconomic indicators (CPI, GDP, consumer confidence)."""
        source = self.data_sources.get("macro")
        if not source:
            return None
        path = Path(source["path"])
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df[source["date_column"]] = pd.to_datetime(df[source["date_column"]])
        return df

    def _merge_datasets(self, sales, promotions, weather, macro) -> pd.DataFrame:
        """Merge all data sources on date and location keys."""
        merged = sales.copy()
        sales_date = self.data_sources["sales"]["date_column"]

        if promotions is not None:
            merged = merged.merge(promotions, on=["sku_id", "location_id"],
                                  how="left", suffixes=("", "_promo"))
            promo_date = self.data_sources["promotions"]["date_column"]
            merged["is_promo"] = (
                (merged[sales_date] >= merged[promo_date]) &
                (merged[sales_date] <= merged.get("promo_end_date", merged[promo_date]))
            ).astype(int)

        if weather is not None:
            weather_date = self.data_sources["weather"]["date_column"]
            merged = merged.merge(weather, left_on=[sales_date, "region"],
                                  right_on=[weather_date, "station_region"], how="left")

        if macro is not None:
            merged["year_month"] = merged[sales_date].dt.to_period("M")
            macro["year_month"] = macro[self.data_sources["macro"]["date_column"]].dt.to_period("M")
            merged = merged.merge(macro, on="year_month", how="left")

        return merged

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to configured frequency and granularity level."""
        date_col = self.data_sources["sales"]["date_column"]
        freq_map = {"daily": "D", "weekly": "W-MON", "monthly": "MS"}
        pd_freq = freq_map.get(self.processing["frequency"], "W-MON")

        group_cols = self.processing["aggregation_level"].split("_")
        df["period"] = df[date_col].dt.to_period(pd_freq[0])

        agg_dict = {"quantity_sold": "sum"}
        for col in ["is_promo", "temperature", "precipitation"]:
            if col in df.columns:
                agg_dict[col] = "max" if col == "is_promo" else ("mean" if col == "temperature" else "sum")

        result = df.groupby(group_cols + ["period"]).agg(agg_dict).reset_index()
        result["period"] = result["period"].dt.to_timestamp()
        return result

    def _save_processed(self, df: pd.DataFrame) -> None:
        """Save processed data to parquet with optional partitioning."""
        output_config = self.config["output"]
        output_path = Path(output_config["processed_path"])
        output_path.mkdir(parents=True, exist_ok=True)

        if output_config.get("partition_by"):
            for keys, group in df.groupby(output_config["partition_by"]):
                if isinstance(keys, str):
                    keys = (keys,)
                part_path = output_path
                for col, val in zip(output_config["partition_by"], keys):
                    part_path = part_path / f"{col}={val}"
                part_path.mkdir(parents=True, exist_ok=True)
                group.to_parquet(part_path / "data.parquet", index=False)
        else:
            df.to_parquet(output_path / "processed_demand.parquet", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run demand data ETL pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()
    pipeline = DataPipeline(config_path=args.config)
    result = pipeline.run()
    print(f"Pipeline complete: {len(result):,} records processed")
