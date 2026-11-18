import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    method: str
    anomaly_mask: np.ndarray
    scores: np.ndarray
    threshold: Optional[float] = None
    params: dict = field(default_factory=dict)


class ZScoreDetector:
    """
    Flags readings more than `threshold` standard deviations from the
    rolling mean. Simple and interpretable — good baseline.

    Limitation: assumes the residuals are normally distributed. Electricity
    consumption is right-skewed (spikes are more common than drops), so
    Z-score catches spikes well but may miss sustained low-usage anomalies.
    """

    def __init__(self, threshold: float = 3.0, window: int = 168):
        self.threshold = threshold
        self.window = window

    def fit_predict(self, series: pd.Series) -> DetectionResult:
        shifted = series.shift(1)
        rolling_mean = shifted.rolling(self.window, min_periods=24).mean()
        rolling_std  = shifted.rolling(self.window, min_periods=24).std()

        z_scores = np.abs((series - rolling_mean) / rolling_std.replace(0, np.nan))
        anomalies = (z_scores > self.threshold).fillna(False).values

        return DetectionResult(
            method="zscore",
            anomaly_mask=anomalies,
            scores=z_scores.fillna(0).values,
            threshold=self.threshold,
            params={"window": self.window},
        )


class IQRDetector:
    """
    Uses the interquartile range per hour-of-day bucket.

    Motivation: 3am and 7pm have very different distributions.
    A reading that is normal at 7pm would be extreme at 3am.
    Per-hour IQR avoids flagging normal daily patterns as anomalies.
    """

    def __init__(self, multiplier: float = 2.5):
        self.multiplier = multiplier
        self.bounds_: dict = {}

    def fit(self, df: pd.DataFrame, col: str = "consumption_kwh"):
        for hour, grp in df.groupby("hour"):
            q1 = grp[col].quantile(0.25)
            q3 = grp[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[hour] = {
                "lower": q1 - self.multiplier * iqr,
                "upper": q3 + self.multiplier * iqr,
            }

    def predict(self, df: pd.DataFrame, col: str = "consumption_kwh") -> DetectionResult:
        if not self.bounds_:
            raise RuntimeError("Call fit() before predict().")

        flags = np.zeros(len(df), dtype=bool)
        scores = np.zeros(len(df))

        for i, row in enumerate(df.itertuples()):
            bounds = self.bounds_.get(row.hour, {"lower": -np.inf, "upper": np.inf})
            val = getattr(row, col)
            dist = max(
                val - bounds["upper"],
                bounds["lower"] - val,
                0,
            )
            scores[i] = dist
            if val < bounds["lower"] or val > bounds["upper"]:
                flags[i] = True

        return DetectionResult(
            method="iqr_per_hour",
            anomaly_mask=flags,
            scores=scores,
            params={"multiplier": self.multiplier},
        )


class IsolationForestDetector:
    """
    Isolation Forest for multi-feature anomaly detection.

    Why Isolation Forest over One-Class SVM?
    - Does not require tuning a kernel; scales well to large datasets.
    - Explicitly designed for anomaly detection rather than adapted from
      a classification framework.
    - contamination parameter is set empirically from domain knowledge
      (typical smart meter datasets have 1-3% anomalies).

    Feature set deliberately excludes the raw reading — using only
    derived features (rolling stats, deviations, deltas) so the model
    learns pattern-break rather than absolute magnitude.
    """

    FEATURES = [
        "rolling_mean_24h", "rolling_std_24h",
        "deviation_24h",    "deviation_pct",
        "delta_1h",         "delta_24h",
        "vs_hour_avg",      "is_weekend",
        "hour",             "month",
    ]

    def __init__(self, contamination: float = 0.02, n_estimators: int = 200):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.FEATURES if c in df.columns]
        return df[cols].fillna(0).values

    def fit(self, df: pd.DataFrame):
        X = self.scaler.fit_transform(self._get_X(df))
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> DetectionResult:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = self.scaler.transform(self._get_X(df))
        raw_scores = self.model.score_samples(X)       # more negative = more anomalous
        predictions = self.model.predict(X)            # -1 = anomaly, 1 = normal
        anomaly_mask = (predictions == -1)

        return DetectionResult(
            method="isolation_forest",
            anomaly_mask=anomaly_mask,
            scores=-raw_scores,
            params={
                "contamination": self.contamination,
                "n_features": len(self.FEATURES),
            },
        )


class EnsembleDetector:
    """
    Combines Z-score, IQR, and Isolation Forest by majority vote.

    A point is flagged only when at least 2 of the 3 methods agree.
    This reduces false positives compared to any single method while
    maintaining recall for strong anomalies that all three catch.

    Trade-off: sustained-shift anomalies (subtle but long) may be
    missed by Z-score and IQR if the shift is gradual, so the ensemble
    can under-detect them. Single-method IQR performs better there.
    """

    def __init__(self,
                 zscore_threshold: float = 3.0,
                 iqr_multiplier: float = 2.5,
                 contamination: float = 0.02):
        self.zs  = ZScoreDetector(threshold=zscore_threshold)
        self.iqr = IQRDetector(multiplier=iqr_multiplier)
        self.ifo = IsolationForestDetector(contamination=contamination)
        self.results_: dict = {}

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "consumption_kwh"

        r_zs  = self.zs.fit_predict(df[col])
        self.iqr.fit(df, col)
        r_iqr = self.iqr.predict(df, col)
        self.ifo.fit(df)
        r_ifo = self.ifo.predict(df)

        self.results_ = {
            "zscore": r_zs,
            "iqr": r_iqr,
            "isolation_forest": r_ifo,
        }

        votes = (
            r_zs.anomaly_mask.astype(int)
            + r_iqr.anomaly_mask.astype(int)
            + r_ifo.anomaly_mask.astype(int)
        )

        df = df.copy()
        df["anomaly_zscore"]    = r_zs.anomaly_mask
        df["anomaly_iqr"]       = r_iqr.anomaly_mask
        df["anomaly_iforest"]   = r_ifo.anomaly_mask
        df["anomaly_votes"]     = votes
        df["anomaly_ensemble"]  = votes >= 2
        df["iforest_score"]     = r_ifo.scores
        df["zscore_score"]      = r_zs.scores

        return df

    def summary(self, df: pd.DataFrame) -> dict:
        total = len(df)
        return {
            "total_readings": total,
            "zscore_flags":        int(df["anomaly_zscore"].sum()),
            "iqr_flags":           int(df["anomaly_iqr"].sum()),
            "iforest_flags":       int(df["anomaly_iforest"].sum()),
            "ensemble_flags":      int(df["anomaly_ensemble"].sum()),
            "ensemble_pct":        round(df["anomaly_ensemble"].mean() * 100, 3),
        }