"""Does the slope+vol architecture add MORE value on single stocks than on ETFs?

Premise to test: single names crash harder, so there is more to buffer.
Counter-premise: much of that extra risk arrives as overnight GAPS, which a
trend/vol system cannot dodge.

SURVIVORSHIP CAVEAT: these are names that still trade today, so ABSOLUTE
returns are upward-biased. We therefore judge the RELATIVE improvement
(strategy vs buy&hold of the SAME names), which is far less contaminated,
and compare it with the ETF result (0.76 -> 0.99 Sharpe, -46% -> -12% DD).
Deliberately includes survivors of catastrophic drawdowns (GE, C, BAC, F,
INTC, BA) to blunt the winner-picking bias.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

STOCKS = ["AAPL","MSFT","JNJ","PG","KO","XOM","JPM","WMT","DIS","INTC",
          "CSCO","PFE","MRK","VZ","T","HD","MCD","NKE","CAT","BA",
          "GE","F","C","BAC","AMD","NVDA","ORCL","IBM","QCOM","TXN"]
ETFS = ["SPY","QQQ","IWM","DIA","EFA","GLD","TLT","VTI","XLF","VNQ"]
COST, START = 0.001, "2006-01-31"
TARGET_VOL, CAP, LB = 0.15, 1.0, 12
ANN = np.sqrt(252)


def load(syms):
    px = yf.download(syms, period="max", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(axis=1, how="all")
    return px


def stats(r):
    r = r.dropna()
    if len(r) < 24:
        return (np.nan,) * 4
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    return cagr, vol, (r.mean() * 12) / vol if vol > 0 else np.nan, (eq / eq.cummax() - 1).min()


def analyse(px, label):
    dret = px.pct_change()
    m = px.resample("ME").last().dropna(how="all")
    rets = m.pct_change()
    logp = np.log(m)
    direction = (logp.diff().rolling(LB).median() > 0).astype(float)
    ewma = pd.DataFrame({c: (dret[c].dropna().ewm(alpha=0.06).std() * ANN).resample("ME").last()
                         for c in dret.columns})

    def run(pos_raw):
        pos = pos_raw.apply(pd.to_numeric, errors="coerce").astype(float)
        pos = pos.reindex(columns=rets.columns).shift(1)
        turn = pos.diff().abs().fillna(0.0)
        return (pos * rets - COST * turn).mean(axis=1)

    bh = rets.mean(axis=1)
    slope_only = run(direction)
    scale = (TARGET_VOL / ewma.reindex_like(rets)).clip(upper=CAP)
    full = run(direction * scale)
    win = bh.loc[START:].dropna().index

    print(f"\n=== {label} ({len(px.columns)} names) ===")
    print(f"{'strategy':28}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}")
    out = {}
    for name, r in [("Buy & Hold", bh), ("Slope only", slope_only), ("Slope x vol (EWMA)", full)]:
        c, v, s, dd = stats(r.loc[win])
        out[name] = (s, dd)
        print(f"{name:28}{c*100:>7.1f}%{v*100:>7.1f}%{s:>8.2f}{dd*100:>8.1f}%")
    ds = out["Slope x vol (EWMA)"][0] - out["Buy & Hold"][0]
    dd_cut = out["Buy & Hold"][1] - out["Slope x vol (EWMA)"][1]
    print(f"-> improvement: Sharpe {ds:+.2f}, drawdown reduced by {dd_cut*100:.1f} pts")

    # how much of the pain arrives as overnight gaps (unhedgeable)?
    gaps = (px / px.shift(1) - 1)
    worst = gaps.min()
    print(f"-> worst single-day move, median across names: {worst.median()*100:.1f}% "
          f"(worst of all: {worst.min()*100:.1f}%)")
    return ds, dd_cut


print("[STOCKS] downloading...", flush=True)
s_impr = analyse(load(STOCKS), "SINGLE STOCKS")
e_impr = analyse(load(ETFS), "ETFs (reference)")

print("\n--- VERDICT ---")
print(f"Sharpe improvement  : stocks {s_impr[0]:+.2f}  vs  ETFs {e_impr[0]:+.2f}")
print(f"Drawdown reduction  : stocks {s_impr[1]*100:.1f} pts  vs  ETFs {e_impr[1]*100:.1f} pts")
print("STOCKS_DONE")
