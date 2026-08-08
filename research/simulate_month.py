"""Walk-forward one-month paper-trade of the calibrated Kronos ETF model.

Strategy (no lookahead):
  * Universe = the 7 trained ETFs.
  * Each trading day t, for each ETF, run the SAME calibrated pipeline the app
    uses and read the calibrated expected return to the 20-day horizon.
  * Signal -> target state with hysteresis (reduces churn/cost):
        buy  if cal_return > +0.2%
        sell if cal_return < -0.2%
        else keep previous state
  * Diversify: equal weight across all ETFs currently in "buy". Rest in cash (0%).
  * Rebalance daily; charge COST on turnover. Earn the ACTUAL t->t+1 return.
Benchmarks: equal-weight buy&hold of the 7, and SPY buy&hold.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from kronos_etf.asset_config import get_asset_config
from kronos_etf.calibration import apply_calibration, load_calibration
from kronos_etf.kronos_utils import (
    choose_best_prediction,
    fetch_etf_klines,
    load_kronos_predictor,
    make_future_timestamps,
)

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "GLD", "TLT"]
SIM_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 21
COST = 0.0010          # 0.10% per unit of turnover (buy or sell)
CAP0 = 10000.0
BUY_TH, SELL_TH = 0.002, -0.002   # hysteresis band on the horizon return


def log(m):
    print(m, flush=True)


def main():
    cfg = get_asset_config("SPY", "etf")
    cal = load_calibration(PROJECT / "finetune/checkpoints_etf/calibration_etf.json")
    log(f"[SIM] calibration w_model={cal.w_model} w_drift={cal.w_drift}")
    predictor = load_kronos_predictor(use_finetuned=True, checkpoint_path=cfg.checkpoint_path)
    lookback, pred_len = cfg.lookback, cfg.pred_len

    data = {}
    for s in SYMBOLS:
        df = fetch_etf_klines(s, period="3y", interval=cfg.interval)
        df = df.dropna(subset=list(cfg.feature_columns)).reset_index(drop=True)
        data[s] = df
        log(f"[SIM] {s}: {len(df)} bars, last {df['timestamp'].iloc[-1].date()}")

    common = sorted(set.intersection(*[set(df["timestamp"]) for df in data.values()]))
    common = list(pd.to_datetime(pd.Series(common)).sort_values())
    N = len(common)
    # decision indices: decide at close of day i, earn return i -> i+1
    decision_idx = list(range(N - 1 - SIM_DAYS, N - 1))
    idx = {s: {t: k for k, t in enumerate(df["timestamp"])} for s, df in data.items()}
    close = {s: df["close"].to_numpy(dtype=float) for s, df in data.items()}

    log(f"[SIM] period: {common[decision_idx[0]].date()} -> {common[decision_idx[-1]+1].date()} "
        f"({SIM_DAYS} decisions)")

    cap = CAP0
    ew_cap = CAP0            # equal-weight buy&hold of the 7
    spy_cap = CAP0
    prev_w = {s: 0.0 for s in SYMBOLS}
    state = {s: False for s in SYMBOLS}   # currently invested?
    n_trades = 0
    days_in_market = 0
    equity = []

    for di in decision_idx:
        d = common[di]
        signals = {}
        for s in SYMBOLS:
            k = idx[s][d]
            x = data[s].iloc[k - lookback + 1 : k + 1]
            x_df = x[list(cfg.feature_columns)].copy()
            x_ts = x["timestamp"].reset_index(drop=True)
            y_ts = make_future_timestamps(x_ts.iloc[-1], pred_len, step=cfg.step, calendar=cfg.calendar)
            pred_df, *_ = choose_best_prediction(predictor, x_df, x_ts, y_ts, pred_len, log=None)
            last_close = float(x_df["close"].iloc[-1])
            band = apply_calibration(pred_df["close"].astype(float), last_close,
                                     x_df["close"].astype(float), cal)
            sig = float(band["close"].iloc[-1]) / last_close - 1.0
            signals[s] = sig
            if sig > BUY_TH:
                state[s] = True
            elif sig < SELL_TH:
                state[s] = False
            # else keep previous state

        invested = [s for s in SYMBOLS if state[s]]
        w = {s: (1.0 / len(invested) if s in invested else 0.0) for s in SYMBOLS}

        turnover = sum(abs(w[s] - prev_w[s]) for s in SYMBOLS)
        cost = COST * turnover * cap
        cap -= cost
        n_trades += sum(1 for s in SYMBOLS if abs(w[s] - prev_w[s]) > 1e-9)
        if invested:
            days_in_market += 1

        # actual next-day returns
        port_ret = 0.0
        for s in SYMBOLS:
            k = idx[s][d]
            r = close[s][k + 1] / close[s][k] - 1.0
            port_ret += w[s] * r
        cap *= (1.0 + port_ret)

        ew_ret = np.mean([close[s][idx[s][d] + 1] / close[s][idx[s][d]] - 1.0 for s in SYMBOLS])
        ew_cap *= (1.0 + ew_ret)
        ks = idx["SPY"][d]
        spy_cap *= (close["SPY"][ks + 1] / close["SPY"][ks])

        equity.append((common[di + 1].date().isoformat(), cap, ew_cap, spy_cap))
        held = ",".join(invested) if invested else "CASH"
        log(f"[{common[di].date()}] sig=" +
            " ".join(f"{s}{signals[s]*100:+.1f}" for s in SYMBOLS) +
            f" | hold={held} | port={cap:.2f}")
        prev_w = w

    log("")
    log("=" * 64)
    log("ONE-MONTH SIMULATION RESULT  (start EUR 10,000)")
    log("=" * 64)
    log(f"PERIOD              : {common[decision_idx[0]].date()} -> {common[decision_idx[-1]+1].date()}")
    log(f"TRADING DAYS        : {SIM_DAYS}")
    log(f"REBALANCE TRADES    : {n_trades}  | days in market: {days_in_market}/{SIM_DAYS}")
    log(f"COST ASSUMPTION     : {COST*100:.2f}% per unit turnover")
    log("")
    log(f"STRATEGY  (net)     : EUR {cap:,.2f}   ({(cap/CAP0-1)*100:+.2f}%)")
    log(f"EQUAL-WEIGHT B&H    : EUR {ew_cap:,.2f}   ({(ew_cap/CAP0-1)*100:+.2f}%)")
    log(f"SPY BUY & HOLD      : EUR {spy_cap:,.2f}   ({(spy_cap/CAP0-1)*100:+.2f}%)")
    log("")
    log(f"STRATEGY vs EW B&H  : {(cap-ew_cap):+,.2f} EUR   ({(cap/ew_cap-1)*100:+.2f}%)")
    log(f"STRATEGY vs SPY B&H : {(cap-spy_cap):+,.2f} EUR   ({(cap/spy_cap-1)*100:+.2f}%)")
    log("=" * 64)
    pd.DataFrame(equity, columns=["date", "strategy", "ew_bh", "spy_bh"]).to_csv(
        PROJECT / "sim_equity.csv", index=False)
    log("SIM_COMPLETE")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"[SIM] wall {time.time()-t0:.0f}s")
