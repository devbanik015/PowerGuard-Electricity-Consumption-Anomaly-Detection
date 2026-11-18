import pandas as pd
import numpy as np


def generate_consumption_data(
    days: int = 365,
    freq: str = "h",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic hourly electricity consumption with:
    - Daily seasonality (peak morning + evening, low overnight)
    - Weekly seasonality (lower weekends)
    - Long-term gradual drift (simulates seasonal change)
    - Gaussian noise
    - Injected anomalies of different types

    Anomaly types injected:
    - Spike: sudden short burst (faulty appliance, surge)
    - Drop: sudden drop to near zero (meter fault, outage)
    - Sustained shift: elevated usage over several days (new equipment)
    """
    rng = np.random.default_rng(seed)
    periods = days * 24
    index = pd.date_range("2024-01-01", periods=periods, freq=freq)

    # Base daily pattern — two peaks (morning 7-9am, evening 6-9pm)
    hour = index.hour
    daily_pattern = (
        3.0 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)   # morning peak
        + 4.5 * np.exp(-0.5 * ((hour - 20) / 2.0) ** 2) # evening peak
        + 1.0                                              # overnight base
    )

    # Weekly effect — weekends roughly 20% lower
    weekly_factor = np.where(index.dayofweek >= 5, 0.80, 1.0)

    # Seasonal drift — higher in winter months
    seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (index.dayofyear / 365) + np.pi)

    # Combine base signal
    base = daily_pattern * weekly_factor * seasonal
    noise = rng.normal(0, 0.3, size=periods)
    consumption = np.clip(base + noise, 0.1, None)

    anomaly_label = np.zeros(periods, dtype=int)

    # --- Inject anomalies ---

    # Type 1: Spikes (10 random hours)
    spike_idx = rng.choice(periods, size=10, replace=False)
    consumption[spike_idx] += rng.uniform(8, 15, size=10)
    anomaly_label[spike_idx] = 1

    # Type 2: Drops (5 random hours)
    drop_idx = rng.choice(periods, size=5, replace=False)
    consumption[drop_idx] = rng.uniform(0.0, 0.3, size=5)
    anomaly_label[drop_idx] = 1

    # Type 3: Sustained shift — 3 day block of elevated usage
    shift_start = rng.integers(1000, periods - 150)
    consumption[shift_start : shift_start + 72] *= rng.uniform(2.0, 2.8)
    anomaly_label[shift_start : shift_start + 72] = 1

    return pd.DataFrame({
        "timestamp": index,
        "consumption_kwh": consumption.round(3),
        "is_anomaly": anomaly_label,
    })


if __name__ == "__main__":
    df = generate_consumption_data()
    df.to_csv("data/raw/consumption.csv", index=False)
    print(f"Generated {len(df):,} rows | "
          f"Anomalies: {df['is_anomaly'].sum()} "
          f"({df['is_anomaly'].mean()*100:.2f}%)")