from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BAND_CFG, N_TESTS, make_forecast_samples, median_forecast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:          # run without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kronos_etf.asset_config import ETF_FINETUNE_UNIVERSE, get_asset_config
from kronos_etf.calibration import CalibrationParams, realized_daily_vol, save_calibration
from kronos_etf.kronos_utils import (
    fetch_etf_klines,
    load_kronos_predictor,
    make_future_timestamps,
    status,
)


DEFAULT_OUTPUT = Path("finetune/checkpoints_etf/calibration_etf.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit leakage-free bias, forecast-combination, and conformal calibration."
    )
    parser.add_argument("--asset-class", choices=["etf"], default="etf")
    parser.add_argument("--symbols", default=",".join(ETF_FINETUNE_UNIVERSE))
    parser.add_argument("--n-windows", type=int, default=15)
    parser.add_argument(
        "--use-finetuned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the asset-class fine-tuned checkpoint (default: true).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    """Split-conformal empirical quantile with finite-sample correction."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Cannot calibrate a quantile from no finite residuals.")
    level = min(1.0, math.ceil((clean.size + 1) * coverage) / clean.size)
    try:
        return float(np.quantile(clean, level, method="higher"))
    except TypeError:  # numpy < 1.22
        return float(np.quantile(clean, level, interpolation="higher"))


def _fit_combination(
    forecast_returns: np.ndarray,
    actual_returns: np.ndarray,
    drift_returns: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, float]:
    corrected = forecast_returns - bias[None, :]
    best_mae = float("inf")
    best_weights = (0.0, 0.0)
    for w_model_i in range(21):
        w_model = w_model_i * 0.05
        for w_drift_i in range(7):
            w_drift = w_drift_i * 0.05
            if w_model + w_drift > 1.0 + 1e-12:
                continue
            combined = w_model * corrected + w_drift * drift_returns
            mae = float(np.mean(np.abs(combined - actual_returns)))
            if mae < best_mae:
                best_mae = mae
                best_weights = (round(w_model, 10), round(w_drift, 10))
    return best_weights


def fit_calibration(
    forecast_returns: np.ndarray,
    actual_returns: np.ndarray,
    drift_returns: np.ndarray,
    sigmas: np.ndarray,
    *,
    asset_class: str,
    checkpoint: str,
) -> CalibrationParams:
    forecast_returns = np.asarray(forecast_returns, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)
    drift_returns = np.asarray(drift_returns, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if forecast_returns.ndim != 2 or forecast_returns.shape != actual_returns.shape:
        raise ValueError("Forecast and actual returns must be equally shaped 2-D arrays.")
    if drift_returns.shape != actual_returns.shape:
        raise ValueError("Drift returns must match actual returns.")
    if sigmas.shape != (actual_returns.shape[0],):
        raise ValueError("Need one volatility value per calibration window.")
    if not (
        np.isfinite(forecast_returns).all()
        and np.isfinite(actual_returns).all()
        and np.isfinite(drift_returns).all()
        and np.isfinite(sigmas).all()
        and (sigmas > 0).all()
    ):
        raise ValueError("Calibration inputs must be finite and volatility must be positive.")

    bias = np.mean(forecast_returns - actual_returns, axis=0)
    w_model, w_drift = _fit_combination(
        forecast_returns, actual_returns, drift_returns, bias
    )
    combined = (
        w_model * (forecast_returns - bias[None, :])
        + w_drift * drift_returns
    )
    scaled_errors = np.abs(actual_returns - combined) / sigmas[:, None]
    q80 = [
        _finite_sample_quantile(scaled_errors[:, h], 0.80)
        for h in range(actual_returns.shape[1])
    ]
    q90 = [
        _finite_sample_quantile(scaled_errors[:, h], 0.90)
        for h in range(actual_returns.shape[1])
    ]
    return CalibrationParams(
        asset_class=asset_class,
        checkpoint=checkpoint,
        pred_len=actual_returns.shape[1],
        n_points=actual_returns.shape[0],
        bias=bias.tolist(),
        w_model=w_model,
        w_drift=w_drift,
        q80=q80,
        q90=q90,
    )


def collect_symbol_windows(
    predictor,
    symbol: str,
    n_windows: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    config = get_asset_config(symbol, "etf")
    lookback = config.lookback
    pred_len = config.pred_len
    years_needed = max(
        7,
        int((lookback + pred_len * (N_TESTS + n_windows + 1)) / 252) + 2,
    )
    status(f"[CAL] {symbol}: loading {years_needed}y of daily history")
    df = fetch_etf_klines(symbol, period=f"{years_needed}y", interval=config.interval)
    min_required = lookback + pred_len * (N_TESTS + n_windows + 1)
    if len(df) < min_required:
        raise ValueError(
            f"{symbol}: need at least {min_required} bars for calibration, got {len(df)}."
        )

    earliest_test_cutoff = len(df) - pred_len - N_TESTS * pred_len
    forecasts: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    drifts: list[np.ndarray] = []
    sigmas: list[float] = []
    for k in range(n_windows, 0, -1):
        cutoff = earliest_test_cutoff - k * pred_len
        x_window = df.iloc[cutoff - lookback : cutoff].reset_index(drop=True)
        y_actual = df.iloc[cutoff : cutoff + pred_len].reset_index(drop=True)
        if len(x_window) != lookback or len(y_actual) != pred_len:
            raise RuntimeError(f"{symbol}: invalid calibration window k={k}.")

        x_timestamp = x_window["timestamp"].reset_index(drop=True)
        y_timestamp = make_future_timestamps(
            x_timestamp.iloc[-1],
            pred_len,
            step=config.step,
            calendar=config.calendar,
        )
        samples = make_forecast_samples(
            predictor,
            x_window,
            x_timestamp,
            y_timestamp,
            list(config.feature_columns),
            pred_len,
        )
        prediction = median_forecast(samples)
        last_close = float(x_window["close"].iloc[-1])
        forecast_ret = prediction["close"].astype(float).to_numpy() / last_close - 1.0
        actual_ret = y_actual["close"].astype(float).to_numpy() / last_close - 1.0
        recent_returns = x_window["close"].astype(float).pct_change().dropna().tail(60)
        mu = float(recent_returns.mean()) if not recent_returns.empty else 0.0

        forecasts.append(forecast_ret)
        actuals.append(actual_ret)
        drifts.append(mu * np.arange(1, pred_len + 1, dtype=float))
        sigmas.append(realized_daily_vol(x_window["close"]))
        status(f"[CAL] {symbol}: window {n_windows - k + 1}/{n_windows}")
    return forecasts, actuals, drifts, sigmas


def main() -> None:
    args = parse_args()
    if args.n_windows <= 0:
        raise ValueError("--n-windows must be positive.")
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        raise ValueError("--symbols must contain at least one ticker.")

    reference_config = get_asset_config(symbols[0], args.asset_class)
    checkpoint = reference_config.checkpoint_path
    status(f"[CAL] loading predictor once for {len(symbols)} symbols")
    predictor = load_kronos_predictor(
        use_finetuned=args.use_finetuned,
        checkpoint_path=checkpoint,
    )
    all_forecasts: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []
    all_drifts: list[np.ndarray] = []
    all_sigmas: list[float] = []
    for symbol in symbols:
        forecasts, actuals, drifts, sigmas = collect_symbol_windows(
            predictor, symbol, args.n_windows
        )
        all_forecasts.extend(forecasts)
        all_actuals.extend(actuals)
        all_drifts.extend(drifts)
        all_sigmas.extend(sigmas)

    params = fit_calibration(
        np.stack(all_forecasts),
        np.stack(all_actuals),
        np.stack(all_drifts),
        np.asarray(all_sigmas),
        asset_class=args.asset_class,
        checkpoint=checkpoint if args.use_finetuned else "BASE_MODEL",
    )
    save_calibration(params, args.output)
    print(
        f"Calibration saved: {args.output}\n"
        f"n_points={params.n_points} pred_len={params.pred_len}\n"
        f"bias_mean={np.mean(params.bias):+.6f}\n"
        f"w_model={params.w_model:.2f} w_drift={params.w_drift:.2f} "
        f"w_naive={1.0 - params.w_model - params.w_drift:.2f}\n"
        f"q80=[{min(params.q80):.3f}, {max(params.q80):.3f}] "
        f"q90=[{min(params.q90):.3f}, {max(params.q90):.3f}]",
        flush=True,
    )


if __name__ == "__main__":
    main()
