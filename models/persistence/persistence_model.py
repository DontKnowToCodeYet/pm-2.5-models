"""
Persistence (Naive) Forecast baseline for PM2.5 - Delhi, Anand Vihar (CPCB hourly data)

Model:  y_hat(t + h) = y(t)           (last observed value carried forward h hours)
Also:   seasonal naive  y_hat(t + 24) = y(t)  is the same as h=24 here (daily cycle).

Split:  chronological 80/20 (no shuffling). Persistence has no parameters, so the
        "train" split is only used to report in-sample error for reference.
Gaps:   raw (unimputed) data. Primary scoring uses only pairs where BOTH the input
        y(t) and the target y(t+h) are observed. A forward-fill variant is also reported.
"""
import sys
import numpy as np
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/Delhi_AnandVihar_2025_Aug2026_11pollutants_combined_unimputed.csv"
TARGET = "PM2.5 (µg/m³)"
HORIZONS = [1, 3, 6, 12, 24]   # hours ahead
TRAIN_FRAC = 0.8


def metrics(y, yhat):
    e = y - yhat
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    nz = y > 0
    mape = np.mean(np.abs(e[nz] / y[nz])) * 100
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(MAE=mae, RMSE=rmse, MAPE_pct=mape, R2=r2, n=len(y))


def evaluate(series, h, idx):
    """Persistence forecast for horizon h, scored on rows in idx."""
    pred = series.shift(h)                 # value observed h hours earlier
    df = pd.DataFrame({"y": series, "yhat": pred}).loc[idx].dropna()
    return metrics(df["y"].values, df["yhat"].values), df


def main():
    df = pd.read_csv(PATH, encoding="utf-8-sig")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("Timestamp").sort_index()
    df = df.asfreq("h")                    # enforce a regular hourly grid
    y = df[TARGET]

    split = df.index[int(len(df) * TRAIN_FRAC)]
    train_idx = df.index[df.index < split]
    test_idx = df.index[df.index >= split]

    print(f"Rows: {len(df)} | {df.index.min()} -> {df.index.max()}")
    print(f"Missing PM2.5: {y.isna().sum()} ({y.isna().mean():.1%})")
    print(f"Train: {train_idx.min()} -> {train_idx.max()} ({len(train_idx)} h)")
    print(f"Test : {test_idx.min()} -> {test_idx.max()} ({len(test_idx)} h)\n")

    rows = []
    y_ffill = y.ffill()
    for h in HORIZONS:
        for split_name, idx in [("train", train_idx), ("test", test_idx)]:
            m, _ = evaluate(y, h, idx)
            rows.append({"horizon_h": h, "split": split_name, "gap_handling": "drop", **m})
        # forward-fill variant: fill gaps in the input only, score on observed targets
        pred = y_ffill.shift(h)
        d = pd.DataFrame({"y": y, "yhat": pred}).loc[test_idx].dropna()
        rows.append({"horizon_h": h, "split": "test", "gap_handling": "ffill",
                     **metrics(d["y"].values, d["yhat"].values)})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(res.round(3).to_string(index=False))
    res.round(4).to_csv("results/persistence/persistence_metrics.csv", index=False)

    # Save the 1-hour-ahead test forecasts for later comparison with other models
    _, d1 = evaluate(y, 1, test_idx)
    d1.rename(columns={"y": "PM25_actual", "yhat": "PM25_persistence_t+1"}).to_csv(
        "results/persistence/persistence_test_predictions_1h.csv")
    print("\nSaved results to results/persistence/")


if __name__ == "__main__":
    main()
