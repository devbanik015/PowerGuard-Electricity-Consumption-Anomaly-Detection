import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


class ConsumptionVisualizer:
    """
    All plots use a shared style sheet so the output looks consistent
    across different machines and Matplotlib backends.
    """

    ANOMALY_COLOR  = "#e63946"
    NORMAL_COLOR   = "#457b9d"
    LINE_COLOR     = "#1d3557"
    FILL_COLOR     = "#a8dadc"
    FIGSIZE_WIDE   = (16, 5)
    FIGSIZE_SQUARE = (10, 6)

    def __init__(self, output_dir: str = "outputs/plots"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
        })

    def plot_overview(self, df: pd.DataFrame, save: bool = True):
        """Full consumption timeline with anomalies overlaid."""
        fig, ax = plt.subplots(figsize=self.FIGSIZE_WIDE)

        normal   = df[~df["anomaly_ensemble"]]
        anomaly  = df[df["anomaly_ensemble"]]

        ax.plot(df["timestamp"], df["consumption_kwh"],
                color=self.LINE_COLOR, lw=0.6, alpha=0.7, label="Consumption")
        ax.scatter(anomaly["timestamp"], anomaly["consumption_kwh"],
                   color=self.ANOMALY_COLOR, s=18, zorder=5, label="Anomaly")

        ax.set_xlabel("Date")
        ax.set_ylabel("Consumption (kWh)")
        ax.set_title("Electricity Consumption — Anomaly Overview", fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        fig.autofmt_xdate()
        ax.legend()
        plt.tight_layout()

        if save:
            path = self.out / "overview.png"
            fig.savefig(path)
            print(f"Saved → {path}")
        return fig

    def plot_detector_comparison(self, df: pd.DataFrame,
                                  window_days: int = 14,
                                  save: bool = True):
        """
        Zooms into a 2-week window to compare which methods fired.
        Useful for understanding disagreements between detectors.
        """
        sample = df.iloc[:window_days * 24].copy()
        methods = ["anomaly_zscore", "anomaly_iqr", "anomaly_iforest", "anomaly_ensemble"]
        labels  = ["Z-Score", "IQR (per hour)", "Isolation Forest", "Ensemble (≥2 votes)"]
        colors  = ["#e76f51", "#2a9d8f", "#e9c46a", self.ANOMALY_COLOR]

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

        for ax, col, label, color in zip(axes, methods, labels, colors):
            ax.plot(sample["timestamp"], sample["consumption_kwh"],
                    color=self.LINE_COLOR, lw=0.7, alpha=0.6)
            flagged = sample[sample[col]]
            ax.scatter(flagged["timestamp"], flagged["consumption_kwh"],
                       color=color, s=25, zorder=5, label=label)
            ax.set_ylabel("kWh", fontsize=9)
            ax.legend(loc="upper right", fontsize=9)

        axes[-1].set_xlabel("Date")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.suptitle(f"Detector Comparison — First {window_days} Days",
                     fontweight="bold", y=1.01)
        fig.autofmt_xdate()
        plt.tight_layout()

        if save:
            path = self.out / "detector_comparison.png"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved → {path}")
        return fig

    def plot_hourly_profile(self, df: pd.DataFrame, save: bool = True):
        """Average consumption by hour — shows daily seasonality shape."""
        profile = df.groupby("hour")["consumption_kwh"].agg(["mean", "std"])

        fig, ax = plt.subplots(figsize=self.FIGSIZE_SQUARE)
        ax.plot(profile.index, profile["mean"],
                color=self.LINE_COLOR, lw=2, marker="o", ms=4)
        ax.fill_between(
            profile.index,
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            color=self.FILL_COLOR, alpha=0.4, label="±1 std dev",
        )
        ax.set_xticks(range(0, 24))
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Consumption (kWh)")
        ax.set_title("Average Hourly Consumption Profile", fontweight="bold")
        ax.legend()
        plt.tight_layout()

        if save:
            path = self.out / "hourly_profile.png"
            fig.savefig(path)
            print(f"Saved → {path}")
        return fig

    def plot_anomaly_heatmap(self, df: pd.DataFrame, save: bool = True):
        """
        Hour-of-day vs day-of-week anomaly frequency heatmap.
        Reveals whether anomalies cluster at specific times
        (e.g. late-night spikes every weekend suggest a pattern,
        not random faults).
        """
        df = df.copy()
        df["day_of_week"] = df["timestamp"].dt.day_name()
        pivot = df.groupby(["hour", "day_of_week"])["anomaly_ensemble"].mean().unstack()

        day_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex(columns=day_order)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="Anomaly Rate")

        ax.set_xticks(range(7))
        ax.set_xticklabels(day_order, rotation=30, ha="right")
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Hour of Day")
        ax.set_title("Anomaly Rate Heatmap (Hour × Day)", fontweight="bold")
        plt.tight_layout()

        if save:
            path = self.out / "anomaly_heatmap.png"
            fig.savefig(path)
            print(f"Saved → {path}")
        return fig

    def plot_score_distribution(self, df: pd.DataFrame, save: bool = True):
        """
        Isolation Forest anomaly score distribution.
        The score separates cleanly when the model works well —
        a large overlap means contamination parameter may need tuning.
        """
        fig, ax = plt.subplots(figsize=self.FIGSIZE_SQUARE)

        normal  = df[~df["anomaly_ensemble"]]["iforest_score"]
        flagged = df[df["anomaly_ensemble"]]["iforest_score"]

        ax.hist(normal,  bins=60, alpha=0.6, color=self.NORMAL_COLOR,  label="Normal")
        ax.hist(flagged, bins=30, alpha=0.7, color=self.ANOMALY_COLOR, label="Anomaly")
        ax.set_xlabel("Isolation Forest Anomaly Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution — Normal vs Anomaly", fontweight="bold")
        ax.legend()
        plt.tight_layout()

        if save:
            path = self.out / "score_distribution.png"
            fig.savefig(path)
            print(f"Saved → {path}")
        return fig