from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:          # run without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kronos_etf.asset_config import get_asset_config
from kronos_etf.calibration import apply_calibration, load_calibration
from kronos_etf.kronos_utils import (
    BATCH_MODE,
    DEVICE_INFO,
    MODEL_VERSION,
    choose_best_prediction,
    fetch_binance_klines,
    fetch_etf_klines,
    fetch_funding_rate,
    load_kronos_predictor,
    make_future_timestamps,
    merge_funding_rate,
    status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kronos live forecast demo (crypto or ETF).")
    parser.add_argument("--symbol", default="BTCUSDT", help="Ticker/symbol, e.g. BTCUSDT or SPY")
    parser.add_argument("--asset-class", choices=["crypto", "etf"], default="crypto")
    parser.add_argument(
        "--use-finetuned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a fine-tuned checkpoint. Defaults to True for crypto, False for ETFs.",
    )
    return parser.parse_args()


def _checkpoint_key(value: str | Path) -> str:
    return str(value).replace("\\", "/").rstrip("/").casefold()


def _load_matching_calibration(config, use_finetuned: bool):
    calibration_path = REPO_ROOT / "finetune/checkpoints_etf/calibration_etf.json"
    params = load_calibration(calibration_path)
    if params is None:
        return None, f"keine Datei unter {calibration_path}"
    if not use_finetuned:
        return None, "Demo nutzt nicht den kalibrierten Fine-Tuning-Checkpoint"
    if params.asset_class != config.asset_class:
        return None, f"Assetklasse passt nicht ({params.asset_class} != {config.asset_class})"
    if params.pred_len != config.pred_len:
        return None, f"Prognoselaenge passt nicht ({params.pred_len} != {config.pred_len})"
    if _checkpoint_key(params.checkpoint) != _checkpoint_key(config.checkpoint_path):
        return None, f"Checkpoint passt nicht ({params.checkpoint} != {config.checkpoint_path})"
    return params, None


def main() -> None:
    args = parse_args()
    config = get_asset_config(args.symbol, args.asset_class)
    use_finetuned = args.use_finetuned
    if use_finetuned is None:
        use_finetuned = config.is_crypto

    status(f"Starte Demo... (symbol={config.symbol}, asset_class={config.asset_class})")
    calibration_params = None
    if config.is_etf:
        try:
            calibration_params, calibration_reason = _load_matching_calibration(config, use_finetuned)
        except (OSError, TypeError, ValueError) as exc:
            calibration_reason = f"ungueltige Kalibrierungsdatei: {exc}"
        if calibration_params is None:
            status(f"Kalibrierung nicht aktiv: {calibration_reason}")
    else:
        status("Kalibrierung nicht aktiv: fuer diese Assetklasse nicht konfiguriert")

    if config.is_crypto:
        status("Hole Krypto-Daten von Binance...")
        try:
            df = fetch_binance_klines(symbol=config.symbol, interval=config.interval, limit=400)
        except RuntimeError as e:
            status(f"[FEHLER] Klines konnten nicht geladen werden: {e}")
            return
        status("DataFrame vorbereitet")

        status("Hole Funding Rate von Binance Futures...")
        try:
            fr_df = fetch_funding_rate(symbol=config.symbol, limit=200)
            df = merge_funding_rate(df, fr_df)
        except RuntimeError as e:
            status(f"[WARNUNG] Funding Rate nicht verfügbar, fahre ohne fort: {e}")
            if "funding_rate" not in df.columns:
                df["funding_rate"] = 0.0
        status("Funding Rate eingebettet")
    else:
        status("Hole ETF-Daten von Yahoo Finance...")
        try:
            df = fetch_etf_klines(config.symbol, period="2y", interval=config.interval)
        except RuntimeError as e:
            status(f"[FEHLER] ETF-Daten konnten nicht geladen werden: {e}")
            return
        status("DataFrame vorbereitet")

    status("Lade Modell...")
    predictor = load_kronos_predictor(use_finetuned=use_finetuned, checkpoint_path=config.checkpoint_path)
    status("Predictor initialisiert")

    lookback = config.lookback
    pred_len = config.pred_len
    feature_columns = list(config.feature_columns)
    if len(df) < lookback:
        status(f"[FEHLER] Nur {len(df)} Bars verfügbar, benötige lookback={lookback}.")
        return
    x_df = df.iloc[-lookback:][feature_columns].copy()
    x_timestamp = df.iloc[-lookback:]["timestamp"].reset_index(drop=True)
    y_timestamp = make_future_timestamps(x_timestamp.iloc[-1], pred_len, step=config.step, calendar=config.calendar)
    status(f"Forecast-Fenster vorbereitet: lookback={lookback}, pred_len={pred_len}")

    status("Berechne Forecast...")
    _t0 = time.perf_counter()
    pred_df, best_cfg, best_score, all_predictions, all_scores = choose_best_prediction(
        predictor, x_df, x_timestamp, y_timestamp, pred_len
    )
    _inference_sec = time.perf_counter() - _t0
    status("Forecast berechnet")
    status(
        f"Gewählt: T={best_cfg['T']}, top_k={best_cfg['top_k']}, top_p={best_cfg['top_p']}, samples={best_cfg['sample_count']}, score={best_score:.6f}"
    )

    last_close = float(x_df["close"].iloc[-1])
    calibrated_band = None
    if calibration_params is not None:
        calibrated_band = apply_calibration(
            pred_df["close"].astype(float),
            last_close,
            x_df["close"].astype(float),
            calibration_params,
        )
        pred_df = pred_df.copy()
        pred_df["close"] = calibrated_band["close"].to_numpy()
        status("Kalibrierung aktiv: bias-korrigierter Forecast mit 80%-Konformalband")
    print(pred_df.head(), flush=True)

    baseline_series = pd.Series([last_close] * pred_len, index=pred_df.index)
    mae_model = float((pred_df["close"].astype(float) - baseline_series).abs().mean())
    band_width_pct: float | None = None
    if calibrated_band is not None:
        band_width_pct = float(
            (calibrated_band["hi80"] - calibrated_band["lo80"]).mean() / last_close * 100
        )
    elif all_predictions:
        stats_band_df = pd.concat(
            [pred["close"].rename(f"path_{i}") for i, pred in enumerate(all_predictions)],
            axis=1,
        )
        stats_band_df["min"] = stats_band_df.min(axis=1)
        stats_band_df["max"] = stats_band_df.max(axis=1)
        band_width_pct = float((stats_band_df["max"] - stats_band_df["min"]).mean() / last_close * 100)
    band_asymmetry = float((pred_df["close"].astype(float).mean() - last_close) / last_close * 100)
    if band_asymmetry > 0.3:
        direction_signal = "LONG_BIAS"
    elif band_asymmetry < -0.3:
        direction_signal = "SHORT_BIAS"
    else:
        direction_signal = "NEUTRAL"

    funding_rate_last = "n/a"
    if "funding_rate" in x_df.columns and not x_df["funding_rate"].fillna(0.0).eq(0.0).all():
        funding_rate_last = f"{float(x_df['funding_rate'].iloc[-1]):.6f}"

    print("", flush=True)
    print("=" * 60, flush=True)
    print("KRONOS STATS REPORT", flush=True)
    print("=" * 60, flush=True)
    print(f"SYMBOL             : {config.symbol}", flush=True)
    print(f"ASSET_CLASS        : {config.asset_class}", flush=True)
    print(f"TIMESTAMP_NOW      : {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    print(f"DEVICE             : {DEVICE_INFO}", flush=True)
    print(f"MODEL_VERSION      : {MODEL_VERSION['label']}", flush=True)
    print(f"BATCH_MODE         : {BATCH_MODE['mode']}", flush=True)
    print(f"LAST_CLOSE         : {last_close:.2f}", flush=True)
    print(f"FUNDING_RATE_LAST  : {funding_rate_last}", flush=True)
    print(f"PRED_LEN           : {pred_len}", flush=True)
    print(f"LOOKBACK           : {lookback}", flush=True)
    print(f"BEST_T             : {best_cfg['T']}", flush=True)
    print(f"BEST_TOP_K         : {best_cfg['top_k']}", flush=True)
    print(f"BEST_TOP_P         : {best_cfg['top_p']}", flush=True)
    print(f"BEST_SAMPLES       : {best_cfg['sample_count']}", flush=True)
    print(f"BEST_SCORE         : {best_score:.6f}", flush=True)
    print(f"INFERENCE_SEC      : {_inference_sec:.1f}", flush=True)
    print(f"FORECAST_END_PRICE : {float(pred_df['close'].iloc[-1]):.2f}", flush=True)
    print(f"FORECAST_MEAN      : {float(pred_df['close'].mean()):.2f}", flush=True)
    band_width_text = f"{band_width_pct:.2f}%" if band_width_pct is not None else "n/a"
    print(f"BAND_WIDTH_PCT     : {band_width_text}", flush=True)
    print(f"BAND_ASYMMETRY_PCT : {band_asymmetry:+.2f}%", flush=True)
    print(f"MAE_VS_LASTVALUE   : {mae_model:.2f}", flush=True)
    print(f"SCORE_CONFIGS_TRIED: {len(all_scores)}", flush=True)
    print(f"DIRECTION_SIGNAL   : {direction_signal}", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

    if all_scores:
        import numpy as np

        scores = np.array(all_scores)
        print(
            f"SCORE_DIST  min={scores.min():.6f}  "
            f"median={np.median(scores):.6f}  "
            f"max={scores.max():.6f}  "
            f"std={scores.std():.6f}",
            flush=True,
        )

    status("Speichere Plot...")
    plot_path = (REPO_ROOT / "outputs").joinpath(f"prediction_plot_{config.symbol.lower()}.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_timestamp, x_df["close"], label="History", linewidth=1.5)
    if calibrated_band is not None:
        ax.fill_between(
            y_timestamp,
            calibrated_band["lo80"].to_numpy(),
            calibrated_band["hi80"].to_numpy(),
            color="#ffb366",
            alpha=0.3,
            label="80% Conformal Band",
        )
    elif all_predictions:
        band_df = pd.concat(
            [pred["close"].rename(f"path_{i}") for i, pred in enumerate(all_predictions)],
            axis=1,
        )
        ax.fill_between(
            y_timestamp,
            band_df.min(axis=1).to_numpy(),
            band_df.max(axis=1).to_numpy(),
            color="#ffb366",
            alpha=0.25,
            label="Forecast Range",
        )
    ax.plot(y_timestamp, pred_df["close"], label="Forecast", linewidth=2.0, color="#ff7f0e")
    price_center = x_df["close"].iloc[-1]
    margin = price_center * 0.08
    ax.set_ylim(price_center - margin, price_center + margin)
    ax.set_title(
        f"Kronos {config.symbol} Forecast | T={best_cfg['T']} top_k={best_cfg['top_k']} "
        f"top_p={best_cfg['top_p']} samples={best_cfg['sample_count']} score={best_score:.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    status(f"Plot gespeichert: {plot_path}")
    status("Demo abgeschlossen")


if __name__ == "__main__":
    main()
