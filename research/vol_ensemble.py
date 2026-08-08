"""Does combining Kronos's vol forecast with GARCH beat either alone?
Reads the already-computed vol_race_results.csv (no GPU). Tests simple
averages, an in-sample optimal weight, and an honest leave-one-symbol-out
optimal weight. Combination in vol space."""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
d = pd.read_csv(PROJECT / "vol_race_results.csv").dropna()
real = d["realized"].values
K, G, E = d["kronos_path"].values, d["garch"].values, d["ewma"].values


def m(pred, r=real):
    corr = np.corrcoef(pred, r)[0, 1]
    rmse = np.sqrt(np.mean((pred - r) ** 2))
    p, rr = pred ** 2, r ** 2
    ql = np.mean(rr / p - np.log(rr / p) - 1)
    return corr, rmse, ql


print(f"{len(d)} points | corr(Kronos,GARCH) forecasts = {np.corrcoef(K, G)[0,1]:.2f}")
print(f"corr of ERRORS (Kronos vs GARCH) = {np.corrcoef(K-real, G-real)[0,1]:.2f}  "
      f"(lower = more orthogonal = combo helps more)\n")

print(f"{'method':26}{'corr':>8}{'RMSE':>9}{'QLIKE':>9}")
rows = {
    "Kronos (path)": K,
    "GARCH": G,
    "EWMA": E,
    "Avg Kronos+GARCH (50/50)": 0.5 * K + 0.5 * G,
    "Avg Kronos+GARCH+EWMA": (K + G + E) / 3,
}
for name, pred in rows.items():
    c, r, q = m(pred)
    print(f"{name:26}{c:>8.3f}{r*100:>8.2f}%{q:>9.3f}")

# in-sample optimal weight on Kronos vs GARCH (min QLIKE)
ws = np.arange(0, 1.01, 0.05)
best_is = min(ws, key=lambda w: m(w * K + (1 - w) * G)[2])
c, r, q = m(best_is * K + (1 - best_is) * G)
print(f"\nIn-sample best weight {best_is:.2f}*Kronos -> corr {c:.3f} RMSE {r*100:.2f}% QLIKE {q:.3f}  [in-sample ceiling]")

# honest leave-one-symbol-out: fit weight on 2 symbols, apply to the 3rd
syms = d["sym"].unique()
loso_pred = np.full(len(d), np.nan)
for s in syms:
    tr = d["sym"] != s
    te = d["sym"] == s
    Ktr, Gtr, Rtr = K[tr.values], G[tr.values], real[tr.values]
    w = min(ws, key=lambda w: m(w * Ktr + (1 - w) * Gtr, Rtr)[2])
    loso_pred[te.values] = w * K[te.values] + (1 - w) * G[te.values]
c, r, q = m(loso_pred)
print(f"Leave-one-symbol-out combo      corr {c:.3f} RMSE {r*100:.2f}% QLIKE {q:.3f}  [honest OOS]")

base_best_q = min(m(K)[2], m(G)[2], m(E)[2])
print(f"\nVerdict: LOSO-combo QLIKE {q:.3f} vs best single {base_best_q:.3f} -> "
      f"{'COMBO WINS' if q < base_best_q else 'no improvement'}")
print("ENSEMBLE_DONE")
