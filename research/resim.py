"""Re-simulate the month OFFLINE from already-logged model signals.
No GPU / no model rerun. Two changes vs the original: COST=0 (fees removed)
and a tunable single sell/buy threshold tau (invested if signal >= tau).
Grid-searches tau to find the in-sample best, and reports benchmarks.
"""
import re, sys
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kronos_etf.kronos_utils import fetch_etf_klines

SYMS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "GLD", "TLT"]
CAP0 = 10000.0

# 1) parse signals from the completed run's log
sig_by_day = []   # list of (date, {sym: signal_fraction})
for line in (PROJECT / "sim_month.log").read_text().splitlines():
    m = re.match(r"\[(\d{4}-\d{2}-\d{2})\] sig=(.+?) \| hold", line)
    if not m:
        continue
    date = pd.Timestamp(m.group(1))
    sigs = {s: float(v) / 100.0 for s, v in re.findall(r"([A-Z]+)([+-][\d.]+)", m.group(2))}
    sig_by_day.append((date, sigs))
dates = [d for d, _ in sig_by_day]
print(f"parsed {len(sig_by_day)} decision days: {dates[0].date()} -> {dates[-1].date()}")

# 2) actual closes -> next-day return per symbol at each decision date
close = {}
for s in SYMS:
    df = fetch_etf_klines(s, period="1y", interval="1d").dropna().set_index("timestamp")
    close[s] = df["close"]
tdates = sorted(close["SPY"].index)


def next_ret(s, d):
    idx = tdates.index(d)
    return close[s].loc[tdates[idx + 1]] / close[s].loc[d] - 1.0


def simulate(tau, cost=0.0):
    cap = CAP0
    prev_w = {s: 0.0 for s in SYMS}
    for d, sigs in sig_by_day:
        inv = [s for s in SYMS if sigs.get(s, -9) >= tau]
        w = {s: (1.0 / len(inv) if s in inv else 0.0) for s in SYMS}
        turn = sum(abs(w[s] - prev_w[s]) for s in SYMS)
        cap -= cost * turn * cap
        cap *= 1.0 + sum(w[s] * next_ret(s, d) for s in SYMS)
        prev_w = w
    return cap


# benchmarks
ew = CAP0
spy = CAP0
for d, _ in sig_by_day:
    ew *= 1.0 + sum(next_ret(s, d) for s in SYMS) / len(SYMS)
    spy *= 1.0 + next_ret("SPY", d)

print(f"\nBenchmarks: EqualWeight-B&H {ew:8.2f} ({(ew/CAP0-1)*100:+.2f}%) | "
      f"SPY-B&H {spy:8.2f} ({(spy/CAP0-1)*100:+.2f}%)")

print("\n(a) Original rule, fees REMOVED (tau via hysteresis approx tau=+0.2%):")
orig_nofee = simulate(0.002, cost=0.0)
print(f"    {orig_nofee:8.2f} ({(orig_nofee/CAP0-1)*100:+.2f}%)")

print("\n(b) Grid-search sell/buy threshold tau (fees=0):")
best = (-1, -1)
for tau_bp in range(-100, 301, 10):
    tau = tau_bp / 10000.0
    v = simulate(tau, cost=0.0)
    if v > best[1]:
        best = (tau, v)
    print(f"    tau={tau*100:+5.1f}%  ->  {v:8.2f}  ({(v/CAP0-1)*100:+.2f}%)")
print(f"\nBEST tau = {best[0]*100:+.1f}%  ->  {best[1]:.2f}  ({(best[1]/CAP0-1)*100:+.2f}%)")
print(f"BEST vs EqualWeight-B&H: {best[1]-ew:+.2f} EUR ({(best[1]/ew-1)*100:+.2f}%)")
print(f"BEST vs SPY-B&H       : {best[1]-spy:+.2f} EUR ({(best[1]/spy-1)*100:+.2f}%)")
print("RESIM_DONE")
