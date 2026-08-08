"""Signal = the ETF's own trailing SLOPE (trend), not a model forecast.

User's proposal: use the median slope of the last few years as the base signal
-> long when the trend is up, flat when it is down. This is time-series
momentum (Moskowitz-Ooi-Pedersen). Tested honestly: signal at month t uses only
data through t and is applied to the t+1 return. Costs charged on turnover.

Two slope definitions:
  median : median of trailing monthly log returns (robust, the user's wording)
  ols    : OLS regression slope of log price over the trailing window
"""
import numpy as np
import pandas as pd
import yfinance as yf

SYMS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "GLD", "TLT", "VTI", "XLF", "VNQ"]
COST = 0.001
LOOKBACKS = [12, 24, 36, 60]        # months
START = "2006-01-31"                # common window (VNQ/VTI etc. all live)

print(f"[SLOPE] downloading {len(SYMS)} ETFs...", flush=True)
px = yf.download(SYMS, period="max", auto_adjust=True, progress=False)["Close"]
m = px.resample("ME").last().dropna(how="all")
logp = np.log(m)
rets = m.pct_change()


def slope_median(lb):
    return logp.diff().rolling(lb).median()


def slope_ols(lb):
    x = np.arange(lb)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    return logp.rolling(lb).apply(lambda y: np.dot(xc, y - y.mean()) / denom, raw=True)


def stats(r):
    r = r.dropna()
    if len(r) < 24:
        return (np.nan,) * 4
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    return cagr, vol, (r.mean() * 12) / vol if vol > 0 else np.nan, (eq / eq.cummax() - 1).min()


def run(sig, cost=COST):
    """Long when slope>0 else flat, per ETF; equal-weight the ETF sleeves."""
    pos = (sig > 0).astype(float).shift(1)          # decide at t, hold t+1
    turn = pos.diff().abs().fillna(0.0)
    net = pos * rets - cost * turn
    return net.mean(axis=1)                          # equal-weight portfolio


bh = rets.mean(axis=1)                               # equal-weight buy & hold
win = bh.loc[START:].dropna().index

print(f"\n[SLOPE] window {win[0].date()} -> {win[-1].date()} ({len(win)} months), "
      f"cost {COST*100:.2f}%/turnover\n")
print(f"{'strategy':30}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}")
c, v, s, dd = stats(bh.loc[win])
print(f"{'Buy & Hold (equal weight)':30}{c*100:>7.1f}%{v*100:>7.1f}%{s:>8.2f}{dd*100:>8.1f}%")
bh_sharpe = s
print()

results = {}
for kind, fn in [("median", slope_median), ("ols", slope_ols)]:
    for lb in LOOKBACKS:
        r = run(fn(lb)).loc[win]
        c, v, s, dd = stats(r)
        results[(kind, lb)] = s
        tag = "  <-- beats B&H" if s > bh_sharpe else ""
        print(f"{f'slope {kind} {lb}m':30}{c*100:>7.1f}%{v*100:>7.1f}%{s:>8.2f}{dd*100:>8.1f}%{tag}")

n_beat = sum(1 for s in results.values() if s > bh_sharpe)
print(f"\n[SLOPE] {n_beat}/{len(results)} configurations beat buy&hold on Sharpe "
      f"({'robust' if n_beat >= len(results) * 0.75 else 'NOT robust'})")

# per-ETF detail for the mid configuration
print("\n--- per-ETF, slope median 24m (long/flat vs always long) ---")
sig = slope_median(24)
pos = (sig > 0).astype(float).shift(1)
turn = pos.diff().abs().fillna(0.0)
net = pos * rets - COST * turn
print(f"{'ETF':6}{'B&H CAGR':>10}{'strat CAGR':>12}{'B&H DD':>9}{'strat DD':>10}{'time in mkt':>12}")
for s_ in SYMS:
    if s_ not in rets.columns:
        continue
    a = stats(rets[s_].loc[win]); b = stats(net[s_].loc[win])
    tim = pos[s_].loc[win].mean() * 100
    print(f"{s_:6}{a[0]*100:>9.1f}%{b[0]*100:>11.1f}%{a[3]*100:>8.1f}%{b[3]*100:>9.1f}%{tim:>11.0f}%")
print("\nSLOPE_DONE")
