"""Does combining a momentum sleeve with 60/40 beat pure 60/40 risk-adjusted?
Static blends (monthly rebalanced), correlation, risk-parity weight, best
in-sample weight. Free data only. The honest test of the diversification thread.
"""
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "SPY","QQQ","IWM","DIA","MDY","XLK","XLE","XLF","XLV","XLI","XLP","XLU","XLY","XLB",
    "EFA","EEM","VGK","EWJ","TLT","IEF","LQD","HYG","GLD","SLV","DBC","VNQ","IYR",
]
COST, CAP0, START = 0.001, 10000.0, "2004-01-31"
TARGET_VOL, VT_WIN = 0.10, 6

raw = yf.download(UNIVERSE, period="max", auto_adjust=True, progress=False)["Close"]
m = raw.resample("ME").last()
rets = m.pct_change()


def momentum_returns(lookback=12, skip=1, top_n=8, cost=COST):
    signal = m.shift(skip) / m.shift(lookback + skip) - 1.0
    out = pd.Series(0.0, index=m.index)
    prev_w = pd.Series(0.0, index=m.columns)
    for i in range(len(m.index) - 1):
        s = signal.iloc[i].dropna()
        if len(s) < top_n + 3:
            continue
        winners = s.sort_values(ascending=False).head(top_n).index
        w = pd.Series(0.0, index=m.columns); w[winners] = 1.0 / top_n
        out.iloc[i + 1] = (w * rets.iloc[i + 1]).sum() - cost * (w - prev_w).abs().sum()
        prev_w = w
    return out


def vol_target(r, target=TARGET_VOL, win=VT_WIN, cap=1.0):
    roll = r.rolling(win).std() * np.sqrt(12)
    return ((target / roll).clip(upper=cap).shift(1) * r)


def stats(r):
    r = r.dropna(); n = len(r); eq = (1 + r).cumprod()
    return (eq.iloc[-1] ** (12 / n) - 1, r.std() * np.sqrt(12),
            (r.mean() * 12) / (r.std() * np.sqrt(12)), (eq / eq.cummax() - 1).min(),
            eq.iloc[-1] * CAP0)


mom = vol_target(momentum_returns(top_n=8))          # top8 = robust choice from the sweep
b6040 = 0.6 * rets["SPY"] + 0.4 * rets["TLT"]
win = pd.concat([mom, b6040], axis=1).loc[START:].dropna().index
mom, b6040 = mom.loc[win], b6040.loc[win]

corr = mom.corr(b6040)
vol_m, vol_b = mom.std(), b6040.std()
rp_w = (1 / vol_m) / (1 / vol_m + 1 / vol_b)          # risk-parity weight on momentum

print(f"\n[COMBO] window {win[0].date()} -> {win[-1].date()} ({len(win)} months)")
print(f"[COMBO] corr(Momentum+VT, 60/40) = {corr:+.2f}   (low corr = diversification works)")
print(f"[COMBO] risk-parity weight on momentum = {rp_w*100:.0f}%\n")

print(f"{'Portfolio':28}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'End(10k)':>11}")
def show(name, r):
    c, v, sh, dd, end = stats(r)
    star = "  <-- beats 60/40" if sh > stats(b6040)[2] else ""
    print(f"{name:28}{c*100:>6.1f}%{v*100:>6.1f}%{sh:>8.2f}{dd*100:>7.1f}%{end:>11,.0f}{star}")

show("60/40 (baseline)", b6040)
show("Momentum+VT only", mom)
for w_ in (0.25, 0.40, 0.50, 0.60, 0.75):
    show(f"Blend {int(w_*100)}% Mom / {int((1-w_)*100)}% 60-40", w_ * mom + (1 - w_) * b6040)
show(f"Risk-parity ({rp_w*100:.0f}% Mom)", rp_w * mom + (1 - rp_w) * b6040)

# best in-sample weight (with caveat)
best = max(((w_, stats(w_ * mom + (1 - w_) * b6040)[2]) for w_ in np.arange(0, 1.01, 0.05)),
          key=lambda x: x[1])
print(f"\n[COMBO] best in-sample weight = {best[0]*100:.0f}% Mom -> Sharpe {best[1]:.2f} "
      f"(vs 60/40 {stats(b6040)[2]:.2f}) [in-sample, treat as ceiling not promise]")
print("COMBO_DONE")
