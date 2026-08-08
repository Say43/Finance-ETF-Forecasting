"""Does a DYNAMIC regime tilt between the momentum sleeve and 60/40 beat the
STATIC blend? Uses only leakage-free as-of-t signals (trend + vol regime).
This is the honest proof-of-concept for the whole 'regime timing' idea
(incl. the future Kronos/LLM version). Free data only.

Regime -> weight on momentum sleeve:
  risk-OFF (downtrend / high vol) -> more momentum (crisis alpha)
  risk-ON  (uptrend / low vol)    -> more 60/40    (wins calm bulls)
Signal at month t sets the weight earned in t+1 (no lookahead).
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
W_OFF, W_ON = 0.65, 0.30          # momentum weight in risk-off / risk-on

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
            (r.mean() * 12) / (r.std() * np.sqrt(12)), (eq / eq.cummax() - 1).min())


mom = vol_target(momentum_returns(top_n=8))
b6040 = 0.6 * rets["SPY"] + 0.4 * rets["TLT"]

# leakage-free regime signals (as-of t)
spy = m["SPY"]
trend_on = (spy > spy.rolling(10).mean())                      # Faber 10-month trend
spy_vol = rets["SPY"].rolling(6).std()
vol_on = spy_vol < spy_vol.rolling(36, min_periods=12).median()  # below trailing median vol

def dyn(sig_on):
    w = pd.Series(np.where(sig_on, W_ON, W_OFF), index=m.index).shift(1)  # t -> t+1
    return (w * mom + (1 - w) * b6040)

combos = {
    "60/40 (baseline)": b6040,
    "Static blend 45% Mom": 0.45 * mom + 0.55 * b6040,
    "Dynamic: trend": dyn(trend_on),
    "Dynamic: vol": dyn(vol_on),
    "Dynamic: trend&vol": dyn(trend_on & vol_on),
    "Dynamic: trend|vol": dyn(trend_on | vol_on),
}
win = pd.concat(list(combos.values()) + [mom], axis=1).loc[START:].dropna().index
base_sh = stats(b6040.loc[win])[2]
static_sh = stats((0.45 * mom + 0.55 * b6040).loc[win])[2]

print(f"\n[TILT] window {win[0].date()} -> {win[-1].date()} ({len(win)} months) | "
      f"weights off/on = {W_OFF}/{W_ON}")
print(f"[TILT] bar to beat: static blend Sharpe = {static_sh:.2f} (60/40 = {base_sh:.2f})\n")
print(f"{'Portfolio':26}{'CAGR':>7}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
for name, r in combos.items():
    c, v, sh, dd = stats(r.loc[win])
    tag = "  <-- beats static" if sh > static_sh + 1e-9 else ""
    print(f"{name:26}{c*100:>6.1f}%{v*100:>6.1f}%{sh:>8.2f}{dd*100:>7.1f}%{tag}")
print("\nTILT_DONE")
