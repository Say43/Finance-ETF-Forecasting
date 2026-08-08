"""Architecture test: SLOPE gives direction, a VOL forecast sizes the position.

This is the intended Kronos architecture:
    direction  <- trailing slope (trend)   [what Kronos lacks]
    sizing     <- forward vol forecast     [Kronos's one proven skill]
    result     <- crash buffering

Before spending ~9 GPU-hours on Kronos vol forecasts, test whether ANY good vol
forecast adds value on top of the slope signal. Proxies: realized vol, EWMA,
GARCH(1,1). If the vol layer helps here, Kronos has a legitimate slot and the
GPU run is justified; if it does not, Kronos cannot help either.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

SYMS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "GLD", "TLT", "VTI", "XLF", "VNQ"]
COST, START = 0.001, "2006-01-31"
TARGET_VOL, CAP = 0.15, 1.0        # no leverage: scale down only
LB = 12                            # slope lookback (best from previous test)

print(f"[ARCH] downloading {len(SYMS)} ETFs...", flush=True)
px = yf.download(SYMS, period="max", auto_adjust=True, progress=False)["Close"]
px = px.dropna(axis=1, how="all")            # drop symbols Yahoo failed to deliver
SYMS = [s for s in SYMS if s in px.columns]
print(f"[ARCH] usable: {len(SYMS)} ETFs -> {', '.join(SYMS)}", flush=True)
dret = px.pct_change()
m = px.resample("ME").last().dropna(how="all")
rets = m.pct_change()
logp = np.log(m)

# --- direction: trailing median monthly log return over LB months ---
slope = logp.diff().rolling(LB).median()
direction = (slope > 0).astype(float)

# --- vol forecasts, computed as-of each month end from DAILY returns ---
ann = np.sqrt(252)
rv = dret.resample("ME").std() * ann                       # realized vol of the month
rv63 = dret.rolling(63).std().resample("ME").last() * ann  # trailing 3m realized


def ewma_vol_series(s, lam=0.94):
    v = s.ewm(alpha=1 - lam).std() * ann
    return v.resample("ME").last()


ewma = pd.DataFrame({c: ewma_vol_series(dret[c].dropna()) for c in dret.columns})


def garch_monthly(sym):
    """One-step-ahead GARCH(1,1) vol at each month end (expanding, as-of t)."""
    s = dret[sym].dropna() * 100
    out = {}
    month_ends = s.resample("ME").last().index
    for i, t in enumerate(month_ends):
        hist = s.loc[:t]
        if len(hist) < 500 or i % 1 != 0:
            continue
        try:
            res = arch_model(hist, vol="Garch", p=1, q=1, mean="Zero").fit(disp="off")
            f = res.forecast(horizon=21, reindex=False)
            out[t] = np.sqrt(f.variance.values[-1].mean()) / 100 * ann
        except Exception:
            pass
    return pd.Series(out)


def stats(r):
    r = r.dropna()
    if len(r) < 24:
        return (np.nan,) * 4
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    return cagr, vol, (r.mean() * 12) / vol if vol > 0 else np.nan, (eq / eq.cummax() - 1).min()


def run(pos_raw, cost=COST):
    pos = pos_raw.apply(pd.to_numeric, errors="coerce").astype(float)
    pos = pos.reindex(columns=rets.columns).shift(1)   # decide at t, hold t+1
    turn = pos.diff().abs().fillna(0.0)
    return (pos * rets - cost * turn).mean(axis=1)


bh = rets.mean(axis=1)
win = bh.loc[START:].dropna().index

variants = {
    "Buy & Hold (equal weight)": bh,
    "Slope only (long/flat)": run(direction),
}
for name, volf in [("trailing RV 3m", rv63), ("EWMA", ewma)]:
    scale = (TARGET_VOL / volf.reindex_like(rets)).clip(upper=CAP)
    variants[f"Slope x volscale ({name})"] = run(direction * scale)

print("[ARCH] fitting GARCH per ETF (may take ~1-2 min)...", flush=True)
g = pd.DataFrame({s: garch_monthly(s) for s in SYMS}).apply(pd.to_numeric, errors="coerce")
g.index = pd.to_datetime(g.index)
scale_g = (TARGET_VOL / g.reindex(rets.index).reindex(columns=rets.columns)
           .apply(pd.to_numeric, errors="coerce")).clip(upper=CAP)
variants["Slope x volscale (GARCH)"] = run(direction * scale_g)
# vol scaling WITHOUT direction, to isolate each contribution
variants["Vol-scale only (no slope)"] = run((TARGET_VOL / rv63.reindex_like(rets)).clip(upper=CAP))

print(f"\n[ARCH] window {win[0].date()} -> {win[-1].date()} ({len(win)} months), "
      f"target vol {TARGET_VOL*100:.0f}%, no leverage\n")
print(f"{'strategy':34}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}")
base = {}
for name, r in variants.items():
    c, v, s, dd = stats(r.loc[win])
    base[name] = (s, dd)
    print(f"{name:34}{c*100:>7.1f}%{v*100:>7.1f}%{s:>8.2f}{dd*100:>8.1f}%")

slope_s, slope_dd = base["Slope only (long/flat)"]
print(f"\n[ARCH] bar to beat = slope only: Sharpe {slope_s:.2f}, MaxDD {slope_dd*100:.1f}%")
best = max((k for k in base if "volscale" in k), key=lambda k: base[k][0])
print(f"[ARCH] best vol-scaled variant: {best} -> Sharpe {base[best][0]:.2f}, "
      f"MaxDD {base[best][1]*100:.1f}%")
verdict = ("VOL LAYER HELPS -> Kronos has a legitimate slot"
           if base[best][0] > slope_s else
           "vol layer does NOT improve on slope alone -> Kronos cannot help here")
print(f"[ARCH] VERDICT: {verdict}")
print("ARCH_DONE")
