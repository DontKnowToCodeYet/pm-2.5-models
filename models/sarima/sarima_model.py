"""
SARIMA(p,0,q)(P,1,Q)_24 baseline for hourly PM2.5 - Delhi, Anand Vihar (CPCB data)

Implemented with numpy/scipy only (statsmodels was not installable in the build
environment). Estimation is by Conditional Sum of Squares (CSS), a standard
ARIMA estimator (Box & Jenkins); order is chosen by AIC on the training split.

Model on the seasonally differenced series  w_t = y_t - y_{t-24}:
    phi(B) Phi(B^24) (w_t - mu) = theta(B) Theta(B^24) e_t

Protocol (identical to the Persistence baseline):
  * chronological 80/20 split; parameters estimated on train only
  * test forecasts at every hourly origin t for horizons 1, 3, 6, 12, 24 h,
    using only data up to t (parameters fixed, no refitting)
  * scored only where the input y(t) and the target y(t+h) are both observed
  * missing inputs are forward-filled (causal - never uses future values)
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/Delhi_AnandVihar_2025_Aug2026_11pollutants_combined_unimputed.csv"
OUT = "results/sarima"
TARGET = "PM2.5 (µg/m³)"
S = 24
HORIZONS = [1, 3, 6, 12, 24]
TRAIN_FRAC = 0.8
BURN = 200                     # residuals ignored at the start of the CSS sum
GRID = dict(p=[1, 2, 3], q=[0, 1, 2], P=[0, 1], Q=[0, 1])


# ---------- polynomial helpers ----------
def ar_poly(phi, Phi):
    a = np.r_[1.0, -np.asarray(phi)]
    A = np.zeros(S * len(Phi) + 1); A[0] = 1.0
    for i, c in enumerate(Phi, 1): A[S * i] = -c
    return np.convolve(a, A)


def ma_poly(theta, Theta):
    b = np.r_[1.0, np.asarray(theta)]
    B = np.zeros(S * len(Theta) + 1); B[0] = 1.0
    for i, c in enumerate(Theta, 1): B[S * i] = c
    return np.convolve(b, B)


def roots_ok(poly):
    """All roots of the lag polynomial outside the unit circle."""
    if len(poly) <= 1 or np.allclose(poly[1:], 0): return True
    return np.all(np.abs(np.roots(poly[::-1])) > 1.0001)


def unpack(x, p, q, P, Q):
    i = 0
    mu = x[i]; i += 1
    phi = x[i:i + p]; i += p
    theta = x[i:i + q]; i += q
    Phi = x[i:i + P]; i += P
    Theta = x[i:i + Q]
    return mu, phi, theta, Phi, Theta


def residuals(x, w, order):
    mu, phi, theta, Phi, Theta = unpack(x, *order)
    return lfilter(ar_poly(phi, Phi), ma_poly(theta, Theta), w - mu)


def css(x, w, order):
    mu, phi, theta, Phi, Theta = unpack(x, *order)
    if not (roots_ok(ar_poly(phi, Phi)) and roots_ok(ma_poly(theta, Theta))):
        return 1e30
    e = residuals(x, w, order)[BURN:]
    return float(np.dot(e, e))


def fit(w, order):
    p, q, P, Q = order
    x0 = np.r_[w.mean(), np.full(p, 0.3 / max(p, 1)), np.zeros(q),
               np.zeros(P), np.full(Q, -0.5)]
    best = None
    for method in ("Nelder-Mead", "Powell"):
        r = minimize(css, x0, args=(w, order), method=method,
                     options=dict(maxiter=20000, maxfev=20000, xatol=1e-6, fatol=1e-6))
        if best is None or r.fun < best.fun: best, x0 = r, r.x
    n = len(w) - BURN
    k = 1 + p + q + P + Q + 1           # + sigma^2
    sigma2 = best.fun / n
    aic = n * np.log(sigma2) + 2 * k
    return best.x, sigma2, aic


# ---------- multi-step forecasts from every origin ----------
def forecast_all(x, order, w, e, origins, H):
    """Return dict h -> forecast of w at origin+h, for every origin (vectorised)."""
    mu, phi, theta, Phi, Theta = unpack(x, *order)
    c = -ar_poly(phi, Phi)[1:]          # w_t - mu = sum c_i (w_{t-i}-mu) + sum m_j e_{t-j} + e_t
    m = ma_poly(theta, Theta)[1:]
    pred = {}
    for k in range(1, H + 1):
        val = np.full(len(origins), mu)
        for i, ci in enumerate(c, 1):
            if ci == 0: continue
            past = (w[origins + k - i] - mu) if i >= k else (pred[k - i] - mu)
            val += ci * past
        for j, mj in enumerate(m, 1):
            if mj == 0 or j < k: continue      # future shocks have expectation 0
            val += mj * e[origins + k - j]
        pred[k] = val
    return pred


def metrics(y, yhat):
    err = y - yhat
    nz = y > 0
    return dict(MAE=np.mean(np.abs(err)), RMSE=np.sqrt(np.mean(err ** 2)),
                MAPE_pct=np.mean(np.abs(err[nz] / y[nz])) * 100,
                R2=1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2), n=len(y))


def main():
    t0 = time.time()
    df = pd.read_csv(PATH, encoding="utf-8-sig")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("Timestamp").sort_index().asfreq("h")
    y = df[TARGET].to_numpy(float)
    yf = pd.Series(y).ffill().bfill().to_numpy()          # causal fill (bfill only for the first rows)
    N = len(y)
    n_train = int(N * TRAIN_FRAC)

    w_full = np.full(N, np.nan)
    w_full[S:] = yf[S:] - yf[:-S]                          # seasonal difference
    w_train = w_full[S:n_train]

    # ---- order selection by AIC on train ----
    rows, fits = [], {}
    for p, q, P, Q in itertools.product(*GRID.values()):
        order = (p, q, P, Q)
        x, s2, aic = fit(w_train, order)
        fits[order] = x
        rows.append(dict(order=f"({p},0,{q})({P},1,{Q})[24]", p=p, q=q, P=P, Q=Q,
                         sigma2=s2, AIC=aic))
        print(f"  SARIMA{rows[-1]['order']:<22} AIC={aic:12.1f}")
    grid = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    best = tuple(grid.loc[0, ["p", "q", "P", "Q"]].astype(int))
    x = fits[best]
    mu, phi, theta, Phi, Theta = unpack(x, *best)
    print(f"\nSelected: {grid.loc[0, 'order']}")
    print(f"  mu={mu:.3f} phi={np.round(phi, 4)} theta={np.round(theta, 4)} "
          f"Phi={np.round(Phi, 4)} Theta={np.round(Theta, 4)}")

    # ---- residuals over the full series with train parameters (no refit) ----
    w0 = np.nan_to_num(w_full, nan=0.0)
    e = np.zeros(N)
    e[S:] = residuals(x, w0[S:], best)

    H = max(HORIZONS)
    split = n_train
    results = []
    for split_name, lo, hi in [("train", S + BURN, split), ("test", split, N)]:
        origins = np.arange(max(lo - 1, S + 100), hi - 1)
        pw = forecast_all(x, best, w0, e, origins, H)
        for h in HORIZONS:
            tgt = origins + h
            keep = (tgt >= lo) & (tgt < hi)
            o, t = origins[keep], tgt[keep]
            yhat = pw[h][keep] + yf[t - S]                 # undo seasonal difference
            ok = ~np.isnan(y[t]) & ~np.isnan(y[o])          # same scoring rule as persistence
            results.append(dict(horizon_h=h, split=split_name, **metrics(y[t][ok], yhat[ok])))
            if split_name == "test" and h == 1:
                pred1 = pd.DataFrame({"PM25_actual": y[t][ok], "PM25_sarima_t+1": yhat[ok]},
                                     index=df.index[t[ok]])

    res = pd.DataFrame(results)
    pd.set_option("display.width", 140)
    print("\n" + res.round(3).to_string(index=False))

    import os
    os.makedirs(OUT, exist_ok=True)
    res.round(4).to_csv(f"{OUT}/sarima_metrics.csv", index=False)
    grid.round(4).to_csv(f"{OUT}/sarima_order_selection.csv", index=False)
    pred1.index.name = "Timestamp"
    pred1.to_csv(f"{OUT}/sarima_test_predictions_1h.csv")
    with open(f"{OUT}/sarima_params.txt", "w") as f:
        f.write(f"order: {grid.loc[0, 'order']}\nmu: {mu}\nphi: {list(phi)}\ntheta: {list(theta)}\n"
                f"Phi: {list(Phi)}\nTheta: {list(Theta)}\nsigma2: {grid.loc[0, 'sigma2']}\n"
                f"AIC: {grid.loc[0, 'AIC']}\n")
    print(f"\nSaved results to {OUT}/  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
