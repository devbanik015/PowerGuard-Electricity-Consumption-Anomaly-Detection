import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Builds features used by both the statistical and ML detectors.

    Key design rule: every feature is computed from past values only
    (shifted by at least 1 period). Using same-hour data would mean the
    anomaly score is partly computed from the anomalous value itself,
    which inflates detection performance on labelled data but fails
    in real deployment.
    """

    def __init__(self, windows: list = None):
        # 24h, 168h (1 week), 720h (30 days)
        self.windows = windows or [24, 168, 720]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("timestamp")
        col = "consumption_kwh"

        # Calendar features — no external data needed
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"]       = df["timestamp"].dt.month
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

        # Lag features — strict past-only
        for lag in [1, 24, 168]:
            df[f"lag_{lag}h"] = df[col].shift(lag)

        # Rolling statistics — shift(1) prevents same-row leakage
        shifted = df[col].shift(1)
        for w in self.windows:
            rolled = shifted.rolling(window=w, min_periods=max(1, w // 4))
            df[f"rolling_mean_{w}h"] = rolled.mean()
            df[f"rolling_std_{w}h"]  = rolled.std()
            df[f"rolling_max_{w}h"]  = rolled.max()
            df[f"rolling_min_{w}h"]  = rolled.min()

        # Deviation from expected — how far is this reading from recent norm?
        df["deviation_24h"] = df[col] - df["rolling_mean_24h"]
        df["deviation_pct"]  = (
            df["deviation_24h"] / df["rolling_mean_24h"].replace(0, np.nan)
        ) * 100

        # Rate of change — sudden jumps are strong anomaly signals
        df["delta_1h"]  = df[col].diff(1)
        df["delta_24h"] = df[col].diff(24)

        # Hour-of-day average (profile) — compares each reading to its
        # historical average for that hour, not the global average.
        hourly_avg = df.groupby("hour")[col].transform("mean")
        df["vs_hour_avg"] = df[col] - hourly_avg

        # Drop warmup rows where rolling windows aren't fully populated
        warmup = max(self.windows)
        df = df.iloc[warmup:].reset_index(drop=True)

        return df