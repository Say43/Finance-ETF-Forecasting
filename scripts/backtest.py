from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:          # run without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kronos_etf.calibration import CalibrationParams, apply_calibration, load_calibration
from kronos_etf.asset_config import get_asset_config
from kronos_etf.kronos_utils import (
    BATCH_MODE,
    DEVICE_INFO,
    MODEL_VERSION,
    _predict_samples,
    fetch_etf_klines,
    fetch_funding_rate,
    load_kronos_predictor,
    make_future_timestamps,
    merge_funding_rate,
    status,
)


BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "qav",
    "trades",
    "tbbav",
    "tbqav",
    "ignore",
]
BAND_CFG = {"T": 0.9, "top_k": 0, "top_p": 0.92, "sample_count": 20}
DATA_LIMIT_CRYPTO = 2000
FUNDING_LIMIT = 500
N_TESTS = 20
USE_FINETUNED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward Kronos backtest (crypto or ETF).")
    parser.add_argument("--symbol", default="BTCUSDT", help="Ticker/symbol, e.g. BTCUSDT or SPY")
    parser.add_argument("--asset-class", choices=["crypto", "etf"], default="crypto")
    parser.add_argument(
        "--use-finetuned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a fine-tuned checkpoint. Defaults to True for crypto, False for ETFs.",
    )
    parser.add_argument(
        "--calibration",
        default="auto",
        metavar="PATH",
        help=(
            "Calibration JSON to apply. 'auto' uses "
            "finetune/checkpoints_etf/calibration_etf.json for ETFs; "
            "'none' disables calibration (default: auto)."
        ),
    )
    return parser.parse_args()


def _checkpoint_key(checkpoint: str, project_dir: Path) -> str:
    path = Path(checkpoint)
    if not path.is_absolute():
        path = project_dir / path
    return str(path.resolve()).replace("\\", "/").rstrip("/").casefold()


def resolve_calibration(
    calibration_arg: str,
    asset_class: str,
    pred_len: int,
    checkpoint_path: str,
    use_finetuned: bool,
    project_dir: Path,
) -> CalibrationParams | None:
    """Load calibration only when it belongs to the predictor being tested."""
    option = calibration_arg.strip()
    if option.casefold() in {"none", "off", "false", "no"}:
        return None

    if option.casefold() == "auto":
        if asset_class != "etf":
            return None
        calibration_path = project_dir / "finetune/checkpoints_etf/calibration_etf.json"
    else:
        calibration_path = Path(option)
        if not calibration_path.is_absolute():
            calibration_path = project_dir / calibration_path

    try:
        params = load_calibration(calibration_path)
    except (OSError, TypeError, ValueError) as exc:
        status(f"[WARNUNG] Kalibrierung konnte nicht geladen werden ({calibration_path}): {exc}. Raw-only.")
        return None

    if params is None:
        if option.casefold() != "auto":
            status(f"[WARNUNG] Kalibrierungsdatei fehlt: {calibration_path}. Raw-only.")
        return None

    mismatch_reason: str | None = None
    if not use_finetuned:
        mismatch_reason = "Backtest verwendet keinen Fine-tuned-Checkpoint"
    elif _checkpoint_key(params.checkpoint, project_dir) != _checkpoint_key(checkpoint_path, project_dir):
        mismatch_reason = (
            f"Checkpoint '{params.checkpoint}' passt nicht zu '{checkpoint_path}'"
        )
    elif params.asset_class != asset_class:
        mismatch_reason = (
            f"Asset-Klasse '{params.asset_class}' passt nicht zu '{asset_class}'"
        )
    elif params.pred_len != pred_len:
        mismatch_reason = f"pred_len {params.pred_len} passt nicht zu {pred_len}"

    if mismatch_reason:
        status(f"[WARNUNG] Kalibrierung ignoriert: {mismatch_reason}. Raw-only.")
        return None

    status(f"Kalibrierung aktiv: {calibration_path}")
    return params


def fetch_binance_klines_2000(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = DATA_LIMIT_CRYPTO) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be positive.")

    chunks: list[list] = []
    end_time: int | None = None
    remaining = limit
    while remaining > 0:
        chunk_limit = min(remaining, 1000)
        params: dict[str, str | int] = {"symbol": symbol, "interval": interval, "limit": chunk_limit}
        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=30)
        if not response.ok:
            raise RuntimeError(f"Binance klines request failed with HTTP {response.status_code}: {response.text}")

        data = response.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError("Binance returned no kline data.")

        chunks = data + chunks
        remaining -= len(data)
        first_open_time = int(data[0][0])
        end_time = first_open_time - 1
        if len(data) < chunk_limit:
            break

    df = pd.DataFrame(chunks[-limit:], columns=BINANCE_COLUMNS).drop_duplicates(subset=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    price_cols = ["open", "high", "low", "close", "volume"]
    df[price_cols] = df[price_cols].astype(float)
    return df[["timestamp", *price_cols]].reset_index(drop=True)


def make_forecast_samples(
    predictor,
    x_window: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    feature_columns: list[str],
    pred_len: int,
) -> list[pd.DataFrame]:
    return _predict_samples(
        predictor,
        x_window[feature_columns].copy(),
        x_timestamp,
        y_timestamp,
        pred_len,
        BAND_CFG,
        log=None,
    )


def median_forecast(samples: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: pd.concat([sample[col] for sample in samples], axis=1).median(axis=1)
            for col in samples[0].columns
        }
    )


def compute_window_metrics(
    window: int,
    cutoff_timestamp: pd.Timestamp,
    x_window: pd.DataFrame,
    y_actual: pd.DataFrame,
    pred_df: pd.DataFrame,
    samples: list[pd.DataFrame],
    elapsed_sec: float,
    calibrated_prediction: dict[str, pd.Series] | None = None,
) -> dict[str, float | int | str | bool]:
    last_close = float(x_window["close"].iloc[-1])
    actual_close = y_actual["close"].astype(float).reset_index(drop=True)
    forecast_close = pred_df["close"].astype(float).reset_index(drop=True)

    actual_end = float(actual_close.iloc[-1])
    forecast_end = float(forecast_close.iloc[-1])
    actual_mean = float(actual_close.mean())
    forecast_mean = float(forecast_close.mean())

    actual_direction = np.sign(actual_end - last_close)
    forecast_direction = np.sign(forecast_end - last_close)
    direction_correct = bool(actual_direction == forecast_direction)
    baseline_direction_correct = bool(actual_end == last_close)

    abs_forecast = (forecast_close - actual_close).abs()
    abs_baseline = (actual_close - last_close).abs()
    mae_forecast = float(abs_forecast.mean())
    mae_baseline = float(abs_baseline.mean())
    mape_forecast = float((abs_forecast / actual_close.abs()).mean() * 100)
    mape_baseline = float((abs_baseline / actual_close.abs()).mean() * 100)

    band = pd.concat([sample["close"].astype(float).reset_index(drop=True) for sample in samples], axis=1)
    band_low = band.min(axis=1)
    band_high = band.max(axis=1)
    band_coverage = float(((actual_close >= band_low) & (actual_close <= band_high)).mean())

    asymmetry_pct = (forecast_mean - last_close) / last_close * 100
    actual_move_pct = (actual_end - last_close) / last_close * 100

    metrics: dict[str, float | int | str | bool] = {
        "window": window,
        "cutoff_timestamp": cutoff_timestamp,
        "last_close": last_close,
        "actual_end": actual_end,
        "forecast_end": forecast_end,
        "actual_move_pct": actual_move_pct,
        "asymmetry_pct": asymmetry_pct,
        "direction_correct": direction_correct,
        "baseline_direction_correct": baseline_direction_correct,
        "mae_forecast": mae_forecast,
        "mae_baseline": mae_baseline,
        "mape_forecast": mape_forecast,
        "mape_baseline": mape_baseline,
        "band_coverage": band_coverage,
        "elapsed_sec": elapsed_sec,
    }
    if calibrated_prediction is not None:
        cal_close = pd.Series(calibrated_prediction["close"], dtype=float).reset_index(drop=True)
        lo80 = pd.Series(calibrated_prediction["lo80"], dtype=float).reset_index(drop=True)
        hi80 = pd.Series(calibrated_prediction["hi80"], dtype=float).reset_index(drop=True)
        lo90 = pd.Series(calibrated_prediction["lo90"], dtype=float).reset_index(drop=True)
        hi90 = pd.Series(calibrated_prediction["hi90"], dtype=float).reset_index(drop=True)

        if not all(len(series) == len(actual_close) for series in (cal_close, lo80, hi80, lo90, hi90)):
            raise ValueError("Calibrated forecast and intervals must match the actual horizon.")

        cal_abs_forecast = (cal_close - actual_close).abs()
        cal_direction = np.sign(float(cal_close.iloc[-1]) - last_close)
        metrics.update(
            {
                "cal_direction_correct": bool(actual_direction == cal_direction),
                "cal_mae_forecast": float(cal_abs_forecast.mean()),
                "cal_mape_forecast": float((cal_abs_forecast / actual_close.abs()).mean() * 100),
                "cov80": float(((actual_close >= lo80) & (actual_close <= hi80)).mean()),
                "cov90": float(((actual_close >= lo90) & (actual_close <= hi90)).mean()),
                "width80_pct": float(((hi80 - lo80) / last_close).mean() * 100),
            }
        )
    return metrics


def diebold_mariano_mae(loss_differences: pd.Series, lag: int) -> tuple[float, float]:
    """Two-sided DM test using a Bartlett/Newey-West long-run variance."""
    differences = np.asarray(loss_differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n_obs = len(differences)
    if n_obs < 2:
        return float("nan"), float("nan")

    mean_difference = float(differences.mean())
    centered = differences - mean_difference
    nw_lag = min(max(int(lag), 0), n_obs - 1)
    nw_variance = float(np.dot(centered, centered) / n_obs)
    for offset in range(1, nw_lag + 1):
        autocovariance = float(np.dot(centered[offset:], centered[:-offset]) / n_obs)
        bartlett_weight = 1.0 - offset / (nw_lag + 1.0)
        nw_variance += 2.0 * bartlett_weight * autocovariance

    variance_of_mean = max(nw_variance / n_obs, 0.0)
    if variance_of_mean <= np.finfo(float).eps:
        if math.isclose(mean_difference, 0.0, abs_tol=np.finfo(float).eps):
            return 0.0, 1.0
        return math.copysign(float("inf"), mean_difference), 0.0

    dm_stat = mean_difference / math.sqrt(variance_of_mean)
    normal_cdf = 0.5 * (1.0 + math.erf(abs(dm_stat) / math.sqrt(2.0)))
    p_value = max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf)))
    return dm_stat, p_value


def print_report(
    results: list[dict[str, float | int | str | bool]],
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    step: int,
    step_delta: pd.Timedelta,
) -> None:
    if not results:
        print("No completed backtest windows.", flush=True)
        return

    results_df = pd.DataFrame(results)
    windows_tested = len(results_df)
    direction_count = int(results_df["direction_correct"].sum())
    baseline_count = int(results_df["baseline_direction_correct"].sum())
    direction_pct = direction_count / windows_tested * 100
    baseline_pct = baseline_count / windows_tested * 100
    direction_edge = direction_pct - baseline_pct

    mae_forecast_mean = float(results_df["mae_forecast"].mean())
    mae_baseline_mean = float(results_df["mae_baseline"].mean())
    mae_improvement = (mae_baseline_mean - mae_forecast_mean) / mae_baseline_mean * 100 if mae_baseline_mean else 0.0
    mape_forecast_mean = float(results_df["mape_forecast"].mean())
    mape_baseline_mean = float(results_df["mape_baseline"].mean())

    band_coverage_mean = float(results_df["band_coverage"].mean() * 100)
    band_coverage_std = float(results_df["band_coverage"].std(ddof=0) * 100)
    model_asymmetry_mean = float(results_df["asymmetry_pct"].mean())
    actual_move_mean = float(results_df["actual_move_pct"].mean())
    bias_error = model_asymmetry_mean - actual_move_mean
    period_start = pd.Timestamp(results_df["cutoff_timestamp"].iloc[0]) - step_delta * lookback
    period_end = pd.Timestamp(results_df["cutoff_timestamp"].iloc[-1]) + step_delta * (pred_len - 1)

    calibrated_metrics: dict[str, float | int] | None = None
    if "cal_mae_forecast" in results_df.columns:
        cal_direction_count = int(results_df["cal_direction_correct"].sum())
        cal_mae_forecast_mean = float(results_df["cal_mae_forecast"].mean())
        cal_mae_improvement = (
            (mae_baseline_mean - cal_mae_forecast_mean) / mae_baseline_mean * 100
            if mae_baseline_mean
            else 0.0
        )
        dm_stat, dm_pval = diebold_mariano_mae(
            results_df["cal_mae_forecast"] - results_df["mae_baseline"],
            lag=pred_len - 1,
        )
        calibrated_metrics = {
            "direction_count": cal_direction_count,
            "mae_improvement": cal_mae_improvement,
            "mape_forecast": float(results_df["cal_mape_forecast"].mean()),
            "cov80": float(results_df["cov80"].mean() * 100),
            "cov90": float(results_df["cov90"].mean() * 100),
            "width80_pct": float(results_df["width80_pct"].mean()),
            "dm_stat": dm_stat,
            "dm_pval": dm_pval,
        }

    print("=" * 70, flush=True)
    print("BACKTEST REPORT", flush=True)
    print("=" * 70, flush=True)
    print(f"WINDOWS_TESTED       : {windows_tested}", flush=True)
    print(f"PERIOD_START         : {period_start}", flush=True)
    print(f"PERIOD_END           : {period_end}", flush=True)
    print(f"DEVICE               : {DEVICE_INFO}", flush=True)
    print(f"MODEL_VERSION        : {MODEL_VERSION['label']}", flush=True)
    print(f"BATCH_MODE           : {BATCH_MODE['mode']}", flush=True)
    print(f"CANDLES_TOTAL        : {len(df)}", flush=True)
    print(f"LOOKBACK             : {lookback}", flush=True)
    print(f"PRED_LEN             : {pred_len}", flush=True)
    print(f"STEP                 : {step}", flush=True)
    print("", flush=True)
    print("--- DIRECTIONAL ACCURACY ---", flush=True)
    print(f"DIRECTION_CORRECT    : {direction_count}/{windows_tested} ({direction_pct:.1f}%)", flush=True)
    print(f"BASELINE_CORRECT     : {baseline_count}/{windows_tested} ({baseline_pct:.1f}%)", flush=True)
    print(f"DIRECTION_EDGE       : {direction_edge:+.1f}%", flush=True)
    print("", flush=True)
    print("--- PRICE ACCURACY ---", flush=True)
    print(f"MAE_FORECAST_MEAN    : {mae_forecast_mean:.2f}", flush=True)
    print(f"MAE_BASELINE_MEAN    : {mae_baseline_mean:.2f}", flush=True)
    print(f"MAE_IMPROVEMENT      : {mae_improvement:.2f}%", flush=True)
    print(f"MAPE_FORECAST_MEAN   : {mape_forecast_mean:.2f}%", flush=True)
    print(f"MAPE_BASELINE_MEAN   : {mape_baseline_mean:.2f}%", flush=True)
    print("", flush=True)
    print("--- BAND QUALITY ---", flush=True)
    print(f"BAND_COVERAGE_MEAN   : {band_coverage_mean:.2f}%", flush=True)
    print(f"BAND_COVERAGE_STD    : {band_coverage_std:.2f}%", flush=True)
    print("", flush=True)
    print("--- BIAS ANALYSIS ---", flush=True)
    print(f"MODEL_ASYMMETRY_MEAN : {model_asymmetry_mean:+.2f}%", flush=True)
    print(f"ACTUAL_MOVE_MEAN     : {actual_move_mean:+.2f}%", flush=True)
    print(f"BIAS_ERROR           : {bias_error:+.2f}%", flush=True)
    if calibrated_metrics is not None:
        cal_direction_count = int(calibrated_metrics["direction_count"])
        cal_direction_pct = cal_direction_count / windows_tested * 100
        print("", flush=True)
        print("--- CALIBRATED (post-processed) ---", flush=True)
        print(f"CAL_DIRECTION_CORRECT : {cal_direction_count}/{windows_tested} ({cal_direction_pct:.1f}%)", flush=True)
        print(f"CAL_MAE_IMPROVEMENT   : {calibrated_metrics['mae_improvement']:+.2f}%", flush=True)
        print(f"CAL_MAPE_FORECAST     : {calibrated_metrics['mape_forecast']:.2f}%", flush=True)
        print(f"COVERAGE_80           : {calibrated_metrics['cov80']:.1f}%   (nominal 80)", flush=True)
        print(f"COVERAGE_90           : {calibrated_metrics['cov90']:.1f}%   (nominal 90)", flush=True)
        print(f"WIDTH_80_MEAN         : {calibrated_metrics['width80_pct']:.2f}%", flush=True)
        print(f"DM_STAT / DM_PVAL     : {calibrated_metrics['dm_stat']:.2f} / {calibrated_metrics['dm_pval']:.3f}", flush=True)
    print("=" * 70, flush=True)


def main() -> None:
    args = parse_args()
    config = get_asset_config(args.symbol, args.asset_class)
    use_finetuned = args.use_finetuned
    if use_finetuned is None:
        use_finetuned = config.is_crypto

    lookback = config.lookback
    pred_len = config.pred_len
    step = pred_len
    feature_columns = list(config.feature_columns)
    step_delta = pd.Timedelta(config.step)

    project_dir = REPO_ROOT
    results_path = project_dir / f"backtest_results_{config.symbol.lower()}.csv"

    if config.is_crypto:
        status("Lade historische Binance-Daten...")
        df = fetch_binance_klines_2000(symbol=config.symbol, interval=config.interval, limit=DATA_LIMIT_CRYPTO)
        try:
            fr_df = fetch_funding_rate(symbol=config.symbol, limit=FUNDING_LIMIT)
            df = merge_funding_rate(df, fr_df)
        except RuntimeError as exc:
            status(f"[WARNUNG] Funding Rate nicht verfügbar, fahre ohne fort: {exc}")
            df["funding_rate"] = 0.0
    else:
        status("Lade historische ETF-Daten von Yahoo Finance...")
        min_required = lookback + pred_len * (N_TESTS + 1)
        # Yahoo daily history is generous; request enough calendar years to
        # cover the required trading-day count with margin for holidays.
        years_needed = max(2, int(min_required / 252) + 2)
        df = fetch_etf_klines(config.symbol, period=f"{years_needed}y", interval=config.interval)

    min_required = lookback + pred_len * (N_TESTS + 1)
    if len(df) < min_required:
        raise ValueError(f"Need at least {min_required} bars for backtest, got {len(df)}.")

    status("Lade Kronos Predictor...")
    predictor = load_kronos_predictor(use_finetuned=use_finetuned, checkpoint_path=config.checkpoint_path)
    status("Predictor geladen")

    calibration_params = resolve_calibration(
        args.calibration,
        config.asset_class,
        pred_len,
        config.checkpoint_path,
        use_finetuned,
        project_dir,
    )

    results: list[dict[str, float | int | str | bool]] = []
    try:
        for i in range(N_TESTS):
            window_num = i + 1
            cutoff_idx = len(df) - pred_len - (N_TESTS - i) * step
            x_window = df.iloc[cutoff_idx - lookback : cutoff_idx].reset_index(drop=True)
            y_actual = df.iloc[cutoff_idx : cutoff_idx + pred_len].reset_index(drop=True)
            if len(x_window) != lookback or len(y_actual) != pred_len:
                raise RuntimeError(f"Invalid window {window_num}: lookback={len(x_window)}, actual={len(y_actual)}")

            x_timestamp = x_window["timestamp"].reset_index(drop=True)
            y_timestamp = make_future_timestamps(
                x_timestamp.iloc[-1], pred_len, step=config.step, calendar=config.calendar
            )
            expected_start = pd.Timestamp(y_timestamp.iloc[0])
            actual_start = pd.Timestamp(y_actual["timestamp"].iloc[0])
            if abs((actual_start - expected_start).total_seconds()) > 60 * 60 * 24:
                status(f"[WARNUNG] Timestamp mismatch window {window_num}: actual={actual_start}, expected={expected_start}")

            t0 = time.perf_counter()
            samples = make_forecast_samples(predictor, x_window, x_timestamp, y_timestamp, feature_columns, pred_len)
            pred_df = median_forecast(samples)
            elapsed = time.perf_counter() - t0

            calibrated_prediction = None
            if calibration_params is not None:
                calibrated_prediction = apply_calibration(
                    pred_df["close"].astype(float).reset_index(drop=True),
                    float(x_window["close"].iloc[-1]),
                    x_window["close"].astype(float).reset_index(drop=True),
                    calibration_params,
                )

            metrics = compute_window_metrics(
                window_num,
                y_actual["timestamp"].iloc[0],
                x_window,
                y_actual,
                pred_df,
                samples,
                elapsed,
                calibrated_prediction=calibrated_prediction,
            )
            results.append(metrics)

            direction_label = "CORRECT" if metrics["direction_correct"] else "WRONG"
            print(
                f"[WINDOW {window_num}/{N_TESTS}] cutoff={pd.Timestamp(metrics['cutoff_timestamp']).strftime('%Y-%m-%d %H:%M')} | "
                f"last={metrics['last_close']:.2f} | "
                f"actual_end={metrics['actual_end']:.2f} ({metrics['actual_move_pct']:+.2f}%) | "
                f"forecast_end={metrics['forecast_end']:.2f} ({metrics['asymmetry_pct']:+.2f}%) | "
                f"dir={direction_label} | mae={metrics['mae_forecast']:.2f} | "
                f"band_cov={metrics['band_coverage']:.2f} | {elapsed:.1f}s",
                flush=True,
            )
    except KeyboardInterrupt:
        print("", flush=True)
        status("Backtest unterbrochen. Schreibe bisherige Ergebnisse...")

    if results:
        output_columns = [
            "window",
            "cutoff_timestamp",
            "last_close",
            "actual_end",
            "forecast_end",
            "actual_move_pct",
            "asymmetry_pct",
            "direction_correct",
            "baseline_direction_correct",
            "mae_forecast",
            "mae_baseline",
            "mape_forecast",
            "band_coverage",
            "elapsed_sec",
        ]
        if calibration_params is not None:
            output_columns.extend(
                [
                    "cal_direction_correct",
                    "cal_mae_forecast",
                    "cal_mape_forecast",
                    "cov80",
                    "cov90",
                    "width80_pct",
                ]
            )
        pd.DataFrame(results)[output_columns].to_csv(results_path, index=False)
        print_report(results, df, lookback, pred_len, step, step_delta)
        status(f"Saved: {results_path}")
    else:
        print_report(results, df, lookback, pred_len, step, step_delta)


if __name__ == "__main__":
    main()
