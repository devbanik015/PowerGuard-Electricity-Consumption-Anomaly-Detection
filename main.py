import logging
import pandas as pd

from src.preprocessor import ConsumptionPreprocessor
from src.feature_engineering import FeatureEngineer
from src.detector import EnsembleDetector
from src.visualizer import ConsumptionVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(raw_path: str = "data/raw/consumption.csv"):
    # 1. Preprocess
    logger.info("Step 1/4 — Preprocessing")
    prep = ConsumptionPreprocessor()
    raw  = prep.load(raw_path)
    df   = prep.clean(raw)
    prep.save(df, "data/processed/consumption_clean.csv")

    # 2. Feature engineering
    logger.info("Step 2/4 — Feature engineering")
    fe = FeatureEngineer()
    df = fe.build(df)

    # 3. Detection
    logger.info("Step 3/4 — Anomaly detection")
    detector = EnsembleDetector(
        zscore_threshold=3.0,
        iqr_multiplier=2.5,
        contamination=0.02,
    )
    df = detector.fit_predict(df)
    summary = detector.summary(df)
    logger.info("Detection summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    df.to_csv("data/processed/results.csv", index=False)

    # 4. Visualise
    logger.info("Step 4/4 — Generating plots")
    viz = ConsumptionVisualizer()
    viz.plot_overview(df)
    viz.plot_detector_comparison(df)
    viz.plot_hourly_profile(df)
    viz.plot_anomaly_heatmap(df)
    viz.plot_score_distribution(df)

    logger.info("Pipeline complete. Results in data/processed/ and outputs/plots/")
    return df, summary


if __name__ == "__main__":
    from src.generate_data import generate_consumption_data

    # Generate synthetic data if raw file doesn't exist
    import os
    if not os.path.exists("data/raw/consumption.csv"):
        logger.info("Generating synthetic data...")
        df_raw = generate_consumption_data()
        os.makedirs("data/raw", exist_ok=True)
        df_raw.to_csv("data/raw/consumption.csv", index=False)

    df, summary = run_pipeline()