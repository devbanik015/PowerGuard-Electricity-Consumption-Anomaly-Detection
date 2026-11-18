# ⚡ PowerGuard — Electricity Consumption Anomaly Detection

A machine learning system that analyses hourly electricity consumption patterns,
detects unusual behaviour using statistical and ML-based techniques, and produces
diagnostic visualisations to support data-driven energy insights.

Built with Python, Pandas, Scikit-learn, and Matplotlib.

---

## 🧠 Motivation

Undetected anomalies in electricity consumption — faulty meters, appliance faults,
energy theft, or sudden demand shifts — cost money and go unnoticed for weeks.
PowerGuard automates the detection process by combining three independent methods
into an ensemble, reducing false alarms while catching genuine faults.

---

## ⚙️ Features

- **Data ingestion** — accepts CSV or Excel smart meter exports, enforces hourly grid, fills gaps
- **Feature engineering** — lag features, rolling statistics, deviation scores, rate-of-change, hour-of-day profile
- **Three detectors**:
  - Z-Score (rolling window, interpretable baseline)
  - IQR per hour-of-day (accounts for daily distribution shift)
  - Isolation Forest (multivariate, ML-based)
- **Ensemble** — majority vote across all three; flags only when ≥2 methods agree
- **Five visualisations** — overview timeline, detector comparison, hourly profile, heatmap, score distribution

---

## 🗂️ Project Structure

```
powerguard/
├── data/
│   ├── raw/                    # Raw meter exports (not committed)
│   └── processed/              # Cleaned data and results
├── src/
│   ├── __init__.py
│   ├── generate_data.py        # Synthetic data with injected anomalies
│   ├── preprocessor.py         # Cleaning, gap-filling, grid enforcement
│   ├── feature_engineering.py  # Lag, rolling, deviation features
│   ├── detector.py             # ZScore, IQR, IsolationForest, Ensemble
│   └── visualizer.py           # Five Matplotlib plot types
├── outputs/
│   └── plots/                  # Generated charts
├── tests/
│   └── test_detector.py        # Unit tests for all detectors
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── main.py                     # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/powerguard.git
cd powerguard
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the full pipeline**
```bash
python main.py
```

This will generate synthetic data, run all three detectors, print a summary,
and save five plots to `outputs/plots/`.

---

## 📊 Design Decisions

### Why three detectors instead of one?
Each method has a different blind spot. Z-score misses gradual sustained shifts.
IQR misses anomalies that are unusual in multivariate context but not by hour.
Isolation Forest can produce false positives on clean data with wrong contamination.
The ensemble takes a majority vote — a point is flagged only when at least two
methods agree, reducing false positives without sacrificing recall on clear anomalies.

### Why per-hour IQR?
A reading of 0.5 kWh at 3am is normal (overnight low). The same reading at 7pm
is a clear drop anomaly. A global IQR threshold would miss the second case because
the overall distribution accommodates low values. Per-hour bucketing gives each
hour its own expected range.

### Why Isolation Forest over One-Class SVM?
Isolation Forest scales to hundreds of thousands of hourly readings without
kernel tricks or distance calculations. It is designed for anomaly detection
from the ground up, not adapted from a classifier. The contamination parameter
maps directly to domain knowledge about anomaly rate.

### Known limitation
The IQR detector iterates row-by-row using `itertuples()`. This is readable and
correct but slow for datasets over 100k rows. A vectorised per-hour approach
using `groupby` + `transform` would be faster — marked as a TODO in `detector.py`.

---

## 📐 Feature Engineering Rules

All features are computed from **past values only** (shifted by at least 1 period).
Using the current reading to compute the anomaly score for the same reading
would inflate performance on labelled data but fail in real deployment.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📦 Requirements

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
matplotlib==3.9.0
scipy==1.13.1
pytest==8.2.2
```

---

## 🔭 Future Improvements

- [ ] Vectorise IQR detector for large datasets
- [ ] Add LSTM autoencoder for sequence-aware anomaly detection
- [ ] Incorporate weather data as an external feature (temperature drives consumption)
- [ ] Export anomaly report as CSV with timestamps and detector breakdown
- [ ] Add Streamlit dashboard for interactive exploration

---

