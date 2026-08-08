"""Litmus test: does VOL-TARGETED cross-sectional momentum beat plain 60/40
risk-adjusted, on a fair common window, after costs? Plus sub-period and
parameter-robustness checks (no cherry-picking a single lucky config).
Free data only (yfinance). No leverage (exposure capped at 1.0)."""
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "SPY","QQQ","IWM","DIA","MDY",
    "XLK","XLE","XLF","XLV","XLI","XLP","XLU","XLY","XLB",
    "EFA","EEM","VGK","EWJ",
    "TLT","IEF","LQD","HYG",
    "GLD","SLV","DBC",
    "VNQ","IYR",
]
COST = 0.001
CAP0 = 10000.0
START = "2004-01-31"          # common window: all series live, fair comparison
TARGET_VOL = 0.10             # match 60/40's ~10% vol
VT_WIN = 6                    # months of trailing vol

print(f"[XSEC2] downloading {len(UNIVERSE)} ETFs...", flush=True)
raw = yf.download(UNIVERSE, period="max", auto_adjust=True, progress=False)["Close"]
m = raw.resample("ME").last()
rets = m.pct_change()


def momentum_returns(lookback=12, skip=1, top_n=5, cost=COST):
    signal = m.shift(skip) / m.shift(lookback + skip) - 1.0
    idx = m.index
    out = pd.Series(0.0, index=idx)
    prev_w = pd.Series(0.0, index=m.columns)
    for i in range(len(idx) - 1):
        s = signal.iloc[i].dropna()
        if len(s) < top_n + 3:
            continue
        winners = s.sort_values(ascending=False).head(top_n).index
        w = pd.Series(0.0, index=m.columns)
        w[winners] = 1.0 / top_n
        turn = (w - prev_w).abs().sum()
        out.iloc[i + 1] = (w * rets.iloc[i + 1]).sum() - cost * turn
        prev_w = w
    return out


def vol_target(r, target=TARGET_VOL, win=VT_WIN, cap=1.0):
    roll = r.rolling(win).std() * np.sqrt(12)          # vol through t
    scale = (target / roll).clip(upper=cap)
    return (scale.shift(1) * r)                         # apply to t+1 (no lookahead)


def stats(r):
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return cagr, vol, sharpe, dd, eq.iloc[-1] * CAP0


strat = momentum_returns()
strat_vt = vol_target(strat)
spy = rets["SPY"]
b6040 = 0.6 * rets["SPY"] + 0.4 * rets["TLT"]

series = {"Momentum (raw)": strat, "Momentum + VolTarget": strat_vt,
          "SPY B&H": spy, "60/40 (SPY/TLT)": b6040}
# fair identical window
win = pd.concat(series.values(), axis=1).loc[START:].dropna().index
print(f"\n[XSEC2] FAIR window {win[0].date()} -> {win[-1].date()} ({len(win)} months), "
      f"target_vol={TARGET_VOL*100:.0f}%, cost {COST*100:.2f}%\n")
print(f"{'':24}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'End(10k)':>11}")
for name, r in series.items():
    c, v, sh, dd, end = stats(r.loc[win])
    print(f"{name:24}{c*100:>6.1f}%{v*100:>6.1f}%{sh:>8.2f}{dd*100:>7.1f}%{end:>11,.0f}")

print("\n--- SUB-PERIODS (Sharpe / MaxDD) ---")
subs = [("2004-2010", "2004-01-31", "2010-12-31"),
        ("2011-2017", "2011-01-31", "2017-12-31"),
        ("2018-2026", "2018-01-31", "2026-12-31")]
print(f"{'':24}" + "".join(f"{lbl:>16}" for lbl, _, _ in subs))
for name, r in series.items():
    row = f"{name:24}"
    for _, a, b in subs:
        seg = r.loc[a:b].dropna()
        if len(seg) > 6:
            _, _, sh, dd, _ = stats(seg)
            row += f"{sh:>7.2f}/{dd*100:>6.1f}%"
        else:
            row += f"{'n/a':>16}"
    print(row)

print("\n--- ROBUSTNESS: raw momentum Sharpe over (lookback x top_n) ---")
print(f"{'lookback':>10}" + "".join(f"{'top'+str(t):>9}" for t in (3, 5, 8, 12)))
for lb in (6, 9, 12):
    row = f"{lb:>10}"
    for t in (3, 5, 8, 12):
        r = momentum_returns(lookback=lb, top_n=t).loc[win]
        _, _, sh, _, _ = stats(r)
        row += f"{sh:>9.2f}"
    print(row)
print("\nXSEC2_DONE")
