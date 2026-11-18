import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConsumptionPreprocessor:
    """
    Cleans raw smart meter exports.

    Design notes:
    - Missing hours are filled with forward-fill then backward-fill rather
      than zero, because zero looks like an outage (a real anomaly signal)
      and we do not want to inject false positives during cleaning.
    - Negative readings are physically impossible — clipped to NaN then filled.
    - Timestamps are enforced to a fixed hourly grid so downstream rolling
      windows are always over a consistent interval.
    """

    def __init__(self, freq: str = "h"):
        self.freq = freq

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        loaders = {".csv": pd.read_csv, ".xlsx": pd.read_excel}
        loader = loaders.get(path.suffix)
        if not loader:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        df = loader(filepath)
        logger.info(f"Loaded {len(df):,} rows from {path.name}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        bad_dates = df["timestamp"].isna().sum()
        if bad_dates:
            logger.warning(f"Dropping {bad_dates} rows with unparseable timestamps")
            df = df.dropna(subset=["timestamp"])

        # Negative consumption is physically impossible
        neg = (df["consumption_kwh"] < 0).sum()
        if neg:
            logger.warning(f"Replacing {neg} negative readings with NaN")
            df.loc[df["consumption_kwh"] < 0, "consumption_kwh"] = np.nan

        df = df.sort_values("timestamp").drop_duplicates("timestamp")

        # Reindex to a complete, evenly-spaced grid — gaps become NaN
        full_idx = pd.date_range(
            df["timestamp"].min(),
            df["timestamp"].max(),
            freq=self.freq,
        )
        df = df.set_index("timestamp").reindex(full_idx)
        df.index.name = "timestamp"

        gap_count = df["consumption_kwh"].isna().sum()
        if gap_count:
            logger.info(f"Filling {gap_count} gaps (ffill → bfill)")
            df["consumption_kwh"] = (
                df["consumption_kwh"].ffill().bfill()
            )

        return df.reset_index()

    def save(self, df: pd.DataFrame, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data → {filepath}")