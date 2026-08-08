"""Rough (~2 min) test: one model decision at the onset of a broad selloff,
hold through it, compare to buy&hold. Shows per-ETF signal vs actual move."""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kronos_etf.asset_config import get_asset_config
from kronos_etf.calibration import apply_calibration, load_calibration
from kronos_etf.kronos_utils import choose_best_prediction, fetch_etf_klines, load_kronos_predictor, make_future_timestamps

SYMS = ["SPY", "QQQ", "IWM", "DIA", "EFA", "GLD", "TLT"]
DECIDE = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2022-08-31")
END = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2022-09-30")
CAP0 = 10000.0
BUY_TH = 0.002

def log(m): print(m, flush=True)

cfg = get_asset_config("SPY", "etf")
cal = load_calibration(PROJECT / "finetune/checkpoints_etf/calibration_etf.json")
pred = load_kronos_predictor(use_finetuned=True, checkpoint_path=cfg.checkpoint_path)
lb, pl = cfg.lookback, cfg.pred_len

t0 = time.time()
log(f"[BEAR] decision {DECIDE.date()} -> hold to {END.date()}")
sig, ret = {}, {}
for s in SYMS:
    df = fetch_etf_klines(s, period="6y", interval="1d").dropna().reset_index(drop=True)
    df = df.set_index("timestamp")
    hist = df.loc[:DECIDE]
    x = hist.iloc[-lb:]
    x_df = x[list(cfg.feature_columns)].copy().reset_index(drop=True)
    x_ts = pd.Series(x.index)
    y_ts = make_future_timestamps(x_ts.iloc[-1], pl, step=cfg.step, calendar=cfg.calendar)
    pred_df, *_ = choose_best_prediction(pred, x_df, x_ts, y_ts, pl, log=None)
    last = float(x_df["close"].iloc[-1])
    band = apply_calibration(pred_df["close"].astype(float), last, x_df["close"].astype(float), cal)
    sig[s] = float(band["close"].iloc[-1]) / last - 1.0
    c0 = hist["close"].iloc[-1]
    c1 = df.loc[:END, "close"].iloc[-1]
    ret[s] = c1 / c0 - 1.0
    log(f"  {s}: signal {sig[s]*100:+5.2f}%  ->  actual {ret[s]*100:+6.2f}%")

invested = [s for s in SYMS if sig[s] > BUY_TH]
log(f"\n[BEAR] model holds: {invested or ['CASH']}")
strat = (1.0 / len(invested) * sum(ret[s] for s in invested)) if invested else 0.0
ew = sum(ret[s] for s in SYMS) / len(SYMS)
spy = ret["SPY"]
log("")
log("=" * 56)
log(f"BROAD-SELLOFF TEST  {DECIDE.date()} -> {END.date()}  (start 10,000)")
log("=" * 56)
log(f"STRATEGY (cash-aware) : {CAP0*(1+strat):9.2f}  ({strat*100:+.2f}%)")
log(f"EQUAL-WEIGHT B&H      : {CAP0*(1+ew):9.2f}  ({ew*100:+.2f}%)")
log(f"SPY BUY & HOLD        : {CAP0*(1+spy):9.2f}  ({spy*100:+.2f}%)")
log(f"STRATEGY vs EW B&H    : {(strat-ew)*100:+.2f}%")
log("=" * 56)
log(f"[BEAR] wall {time.time()-t0:.0f}s")
log("BEAR_DONE")
