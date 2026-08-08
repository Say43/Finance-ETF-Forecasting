"""Make-or-break test: does cross-sectional momentum over a broad FREE ETF
universe beat 60/40 after costs? No LLM, no Kronos yet - just the return engine.
Monthly rebalance, 12-1 momentum, long-only top-N, transaction costs.
"""
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "SPY","QQQ","IWM","DIA","MDY",          # US size
    "XLK","XLE","XLF","XLV","XLI","XLP","XLU","XLY","XLB",  # US sectors
    "EFA","EEM","VGK","EWJ",                # intl
    "TLT","IEF","LQD","HYG",                # bonds
    "GLD","SLV","DBC",                      # commodities
    "VNQ","IYR",                            # real estate
]
TOP_N = 5
COST = 0.001          # 0.10% per unit turnover
CAP0 = 10000.0
LOOKBACK, SKIP = 12, 1

print(f"[XSEC] downloading {len(UNIVERSE)} ETFs (free, yfinance)...", flush=True)
raw = yf.download(UNIVERSE, period="max", auto_adjust=True, progress=False)["Close"]
m = raw.resample("ME").last()
rets = m.pct_change()
signal = m.shift(SKIP) / m.shift(LOOKBACK + SKIP) - 1.0   # 12-1 momentum
print(f"[XSEC] monthly data: {m.index[0].date()} -> {m.index[-1].date()} ({len(m)} months)", flush=True)


def curve_from_monthly(r):
    return CAP0 * (1 + r.fillna(0)).cumprod()


def stats(r):
    r = r.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return cagr, vol, sharpe, dd, eq.iloc[-1] * CAP0


# strategy: form weights at month i (top-N by signal), earn rets at i+1
idx = m.index
strat_ret = pd.Series(0.0, index=idx)
prev_w = pd.Series(0.0, index=m.columns)
n_valid_start = None
for i in range(len(idx) - 1):
    s = signal.iloc[i].dropna()
    if len(s) < TOP_N + 3:      # need a reasonable cross-section
        continue
    if n_valid_start is None:
        n_valid_start = idx[i]
    winners = s.sort_values(ascending=False).head(TOP_N).index
    w = pd.Series(0.0, index=m.columns)
    w[winners] = 1.0 / TOP_N
    turn = (w - prev_w).abs().sum()
    r_next = (w * rets.iloc[i + 1]).sum() - COST * turn
    strat_ret.iloc[i + 1] = r_next
    prev_w = w

strat_ret = strat_ret.loc[n_valid_start:]

# benchmarks over the same window
bench_spy = rets["SPY"].loc[strat_ret.index]
bench_6040 = (0.6 * rets["SPY"] + 0.4 * rets["TLT"]).loc[strat_ret.index]
avail = rets.loc[strat_ret.index]
bench_ew = avail.mean(axis=1)

print(f"\n[XSEC] backtest window: {strat_ret.index[0].date()} -> {strat_ret.index[-1].date()} "
      f"({len(strat_ret)} months), top-{TOP_N}, cost {COST*100:.2f}%\n")
print(f"{'':22}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'End(10k)':>12}")
for name, r in [("XSec-Momentum (net)", strat_ret), ("SPY B&H", bench_spy),
                ("60/40 (SPY/TLT)", bench_6040), ("Equal-Weight all", bench_ew)]:
    c, v, sh, dd, end = stats(r)
    print(f"{name:22}{c*100:>7.1f}%{v*100:>7.1f}%{sh:>8.2f}{dd*100:>8.1f}%{end:>12,.0f}")
print("\nXSEC_DONE")
