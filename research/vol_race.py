"""Volatility horse-race: does KRONOS's forward-vol forecast beat simple
baselines (persistence, EWMA/RiskMetrics, GARCH(1,1)) at predicting the
realized vol of the next 20 trading days? This tests Kronos's ONE genuine
candidate skill (distribution/uncertainty), not return prediction.

Kronos vol = mean over its sampled forecast paths of each path's daily-return
std, annualized. All forecasts use data up to the cutoff only.

NOTE: the fine-tuned checkpoint saw ETF history in training, so a *historical*
win carries a leakage caveat (would need forward validation). But a *loss*
here is conclusive - it can't beat GARCH even with any such tailwind.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kronos_etf.asset_config import get_asset_config
from kronos_etf.kronos_utils import (_call_predict, _call_predict_batch, fetch_etf_klines,
                                load_kronos_predictor, make_future_timestamps)
from arch import arch_model

SYMBOLS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["SPY", "QQQ", "TLT"]
N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 36
BAND_CFG = {"T": 0.9, "top_k": 0, "top_p": 0.92, "sample_count": 24}
ANN = np.sqrt(252)


def log(m): print(m, flush=True)


def raw_samples(predictor, x_df, x_ts, y_ts, pl, cfg):
    """Model's RAW forecast paths (no align/smooth) so vol isn't distorted."""
    batch = _call_predict_batch(predictor, x_df, x_ts, y_ts, pl, cfg)
    if batch is not None:
        return batch
    return [_call_predict(predictor, x_df, x_ts, y_ts, pl, cfg)
            for _ in range(int(cfg["sample_count"]))]


def kronos_vols(samples, pl):
    """Two fair vol readings from raw paths:
    kv_path = mean within-path daily-return std (annualized)
    kv_disp = cross-sample terminal-return dispersion / sqrt(H) (annualized)."""
    pv, terminals = [], []
    for s in samples:
        cl = s["close"].astype(float).reset_index(drop=True)
        r = cl.pct_change().dropna()
        if len(r) > 1:
            pv.append(r.std())
        if len(cl) > 1 and cl.iloc[0] > 0:
            terminals.append(cl.iloc[-1] / cl.iloc[0] - 1.0)
    kv_path = float(np.mean(pv)) * ANN if pv else np.nan
    kv_disp = (float(np.std(terminals)) / np.sqrt(pl)) * ANN if len(terminals) > 1 else np.nan
    return kv_path, kv_disp


def ewma_vol(returns, lam=0.94):
    var = returns.iloc[0] ** 2
    for x in returns.iloc[1:]:
        var = lam * var + (1 - lam) * x ** 2
    return np.sqrt(var) * ANN


def garch_vol(returns, horizon):
    r = returns.dropna() * 100.0
    am = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    daily_var = fc.variance.values[-1] / (100.0 ** 2)
    return np.sqrt(daily_var.mean()) * ANN


def qlike(pred, real):   # QLIKE loss on variance (proper vol-forecast scoring)
    p, r = pred ** 2, real ** 2
    return float(np.mean(r / p - np.log(r / p) - 1))


def main():
    cfg = get_asset_config("SPY", "etf")
    predictor = load_kronos_predictor(use_finetuned=True, checkpoint_path=cfg.checkpoint_path)
    lb, pl = cfg.lookback, cfg.pred_len
    rows = []
    t0 = time.time()
    for sym in SYMBOLS:
        df = fetch_etf_klines(sym, period="7y", interval="1d").dropna().reset_index(drop=True)
        n = len(df)
        # non-overlapping test windows ending at the most recent data
        cutoffs = [n - pl - k * pl for k in range(N_WINDOWS)][::-1]
        cutoffs = [c for c in cutoffs if c - lb >= 0]
        log(f"[VOL] {sym}: {n} bars, {len(cutoffs)} windows")
        rets_all = df["close"].astype(float).pct_change()
        for ci, c in enumerate(cutoffs):
            x = df.iloc[c - lb:c]
            x_df = x[list(cfg.feature_columns)].copy().reset_index(drop=True)
            x_ts = x["timestamp"].reset_index(drop=True)
            y_ts = make_future_timestamps(x_ts.iloc[-1], pl, step=cfg.step, calendar=cfg.calendar)
            samples = raw_samples(predictor, x_df, x_ts, y_ts, pl, BAND_CFG)
            kv_path, kv_disp = kronos_vols(samples, pl)
            hist = rets_all.iloc[c - lb:c].dropna()
            persist = hist.iloc[-pl:].std() * ANN
            ew = ewma_vol(hist)
            try:
                g = garch_vol(hist, pl)
            except Exception:
                g = np.nan
            realized = rets_all.iloc[c:c + pl].std() * ANN
            rows.append({"sym": sym, "kronos_path": kv_path, "kronos_disp": kv_disp,
                         "persist": persist, "ewma": ew, "garch": g, "realized": realized})
            if (ci + 1) % 10 == 0:
                log(f"  {sym} {ci+1}/{len(cutoffs)} | {time.time()-t0:.0f}s")

    d = pd.DataFrame(rows).dropna()
    log(f"\n[VOL] {len(d)} valid (symbol,window) points\n")
    methods = ["kronos_path", "kronos_disp", "persist", "ewma", "garch"]
    log(f"{'method':13}{'corr':>8}{'spearman':>10}{'RMSE':>9}{'QLIKE':>9}")
    res = {}
    for mth in methods:
        corr = d[mth].corr(d["realized"])
        sp = d[mth].corr(d["realized"], method="spearman")
        rmse = float(np.sqrt(((d[mth] - d["realized"]) ** 2).mean()))
        ql = qlike(d[mth].values, d["realized"].values)
        res[mth] = (corr, sp, rmse, ql)
        log(f"{mth:13}{corr:>8.3f}{sp:>10.3f}{rmse*100:>8.2f}%{ql:>9.3f}")
    baselines = ["persist", "ewma", "garch"]
    best_base_corr = max(res[m][0] for m in baselines)
    best_kronos_corr = max(res["kronos_path"][0], res["kronos_disp"][0])
    log(f"\n[VOL] best correlation: {max(methods, key=lambda x: res[x][0])} | "
        f"best QLIKE: {min(methods, key=lambda x: res[x][3])}")
    log(f"[VOL] kronos best corr {best_kronos_corr:.3f} vs best baseline {best_base_corr:.3f} "
        f"-> {'KRONOS WINS' if best_kronos_corr > best_base_corr else 'baseline wins'}")
    d.to_csv(PROJECT / "vol_race_results.csv", index=False)
    log("VOL_RACE_DONE")


if __name__ == "__main__":
    main()
