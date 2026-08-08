from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "finetune" / "data"

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
KLINE_START = pd.Timestamp("2017-08-17 00:00:00", tz="UTC")
FUNDING_START = pd.Timestamp("2019-09-13 00:00:00", tz="UTC")
WINDOW_SIZE = 360 + 48
WINDOW_STEP = 24

ETF_FINETUNE_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "EFA", "GLD", "TLT"]
ETF_LOOKBACK = 360
ETF_PRED_LEN = 20
ETF_WINDOW_SIZE = ETF_LOOKBACK + ETF_PRED_LEN
ETF_WINDOW_STEP = 5
ETF_PERIOD = "15y"
# ETFs move far less than BTC; crash/bear regime thresholds over a 7-bar
# (7 trading day) window are tightened accordingly (see utils/asset_config.py).
ETF_CRASH_THRESHOLD = -0.07
ETF_BEAR_THRESHOLD = -0.015
CRYPTO_CRASH_THRESHOLD = -0.10
CRYPTO_BEAR_THRESHOLD = -0.02

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "funding_rate",
    "rsi_14",
    "atr_14",
    "bb_width",
    "vol_zscore",
    "log_return",
    "realized_vol",
    "funding_8h_ma",
]
KRONOS_FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]

TRAIN_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
VAL_CUTOFF = pd.Timestamp("2025-10-01", tz="UTC")


def status(message: str) -> None:
    print(f"[PREP] {message}", flush=True)


def require_datasets():
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing package 'datasets'. Install the official Chronos training dependencies: "
            'pip install "chronos-forecasting[training]" gluonts datasets accelerate'
        ) from exc
    return Dataset


def fetch_klines() -> pd.DataFrame:
    rows = []
    start_ms = int(KLINE_START.timestamp() * 1000)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    request_count = 0

    while start_ms < now_ms:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": SYMBOL, "interval": INTERVAL, "limit": 1000, "startTime": start_ms},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"Binance klines request failed with HTTP {response.status_code}: {response.text}")
        data = response.json()
        if not data:
            break
        rows.extend(data)
        request_count += 1
        last_open_time = int(data[-1][0])
        next_start = last_open_time + 60 * 60 * 1000
        if next_start <= start_ms:
            break
        start_ms = next_start
        status(f"Fetched klines: {len(rows)} candles ({request_count} requests)")
        time.sleep(0.2)

    columns = [
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
    df = pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df[["timestamp", *numeric_cols]]


def fetch_funding_rates() -> pd.DataFrame:
    rows = []
    start_ms = int(FUNDING_START.timestamp() * 1000)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    request_count = 0

    while start_ms < now_ms:
        response = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": SYMBOL, "limit": 1000, "startTime": start_ms},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"Binance funding request failed with HTTP {response.status_code}: {response.text}")
        data = response.json()
        if not data:
            break
        rows.extend(data)
        request_count += 1
        last_funding_time = int(data[-1]["fundingTime"])
        next_start = last_funding_time + 1
        if next_start <= start_ms:
            break
        start_ms = next_start
        status(f"Fetched funding rates: {len(rows)} rows ({request_count} requests)")
        time.sleep(0.2)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["fundingTime"])
    df = df.sort_values("fundingTime").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding_rate"]]


def merge_funding(df: pd.DataFrame, fr_df: pd.DataFrame) -> pd.DataFrame:
    if fr_df.empty:
        out = df.copy()
        out["funding_rate"] = 0.0
        return out
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        fr_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged["funding_rate"] = merged["funding_rate"].fillna(0.0).astype(float)
    return merged.reset_index(drop=True)


def fetch_etf_ohlcv(symbol: str, period: str = ETF_PERIOD, interval: str = "1d") -> pd.DataFrame:
    """Free daily OHLCV history for an ETF/equity ticker via Yahoo Finance."""
    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["funding_rate"] = 0.0
    return df[["timestamp", *numeric_cols, "funding_rate"]].reset_index(drop=True)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount"] = out["volume"] * out[["open", "high", "low", "close"]].mean(axis=1)
    out["rsi_14"] = compute_rsi(out["close"], 14)

    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean()

    middle = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_width"] = ((middle + 2 * std) - (middle - 2 * std)) / middle

    vol_mean = out["volume"].rolling(168).mean()
    vol_std = out["volume"].rolling(168).std()
    out["vol_zscore"] = (out["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["realized_vol"] = out["log_return"].rolling(24).std() * math.sqrt(24)
    out["funding_8h_ma"] = out["funding_rate"].rolling(3).mean()
    return out


def add_regimes(df: pd.DataFrame, crash_threshold: float, bear_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["price_change_7d"] = (out["close"] - out["close"].shift(168)) / out["close"].shift(168)
    out["realized_vol_7d"] = out["log_return"].rolling(168).std() * math.sqrt(168)

    conditions = [
        out["price_change_7d"] < crash_threshold,
        out["price_change_7d"] < bear_threshold,
        out["price_change_7d"] > -bear_threshold,
    ]
    choices = ["CRASH", "BEAR", "BULL"]
    out["regime"] = np.select(conditions, choices, default="SIDEWAYS")
    return out


def print_regime_distribution(df: pd.DataFrame) -> None:
    counts = df["regime"].value_counts().reindex(["BULL", "BEAR", "SIDEWAYS", "CRASH"], fill_value=0)
    total = int(counts.sum())
    print("Regime distribution:", flush=True)
    for regime, count in counts.items():
        pct = count / total * 100 if total else 0.0
        print(f"  {regime:8s}: {int(count):7d} ({pct:5.2f}%)", flush=True)


def make_windows(df: pd.DataFrame, window_size: int, step: int, symbol: str = "") -> list[dict]:
    records = []
    for start_idx in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start_idx : start_idx + window_size]
        regime = str(window["regime"].iloc[-1])
        record = {
            "start": window["timestamp"].iloc[0].isoformat(),
            "features": window[KRONOS_FEATURE_COLUMNS].astype("float32").to_numpy().tolist(),
            "target": window["close"].astype("float32").to_numpy().tolist(),
            "regime": regime,
        }
        if symbol:
            record["symbol"] = symbol
        records.append(record)
    return records


def make_windows_by_target_split(
    df: pd.DataFrame,
    window_size: int,
    lookback: int,
    step: int,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
    symbol: str = "",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Slide a window across the full (unsplit) timeline and bucket each
    window into train/val/test by its *target* (prediction) start time only.

    Splitting by requiring the whole window to sit inside a date-sliced
    dataframe breaks down when the split period is shorter than the window
    (e.g. a 9-month val period vs. a 380-day ETF window) - it silently
    yields zero windows. Bucketing by target start avoids that: context
    may reach back before the split boundary (same as a live backtest,
    which always conditions on real historical prices), but no window's
    prediction target crosses into another split.
    """
    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []
    for start_idx in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start_idx : start_idx + window_size]
        target_start = window["timestamp"].iloc[lookback]
        regime = str(window["regime"].iloc[-1])
        record = {
            "start": window["timestamp"].iloc[0].isoformat(),
            "features": window[KRONOS_FEATURE_COLUMNS].astype("float32").to_numpy().tolist(),
            "target": window["close"].astype("float32").to_numpy().tolist(),
            "regime": regime,
        }
        if symbol:
            record["symbol"] = symbol

        if target_start < train_cutoff:
            train_records.append(record)
        elif target_start < val_cutoff:
            val_records.append(record)
        else:
            test_records.append(record)
    return train_records, val_records, test_records


def balance_train_windows(records: list[dict]) -> tuple[list[dict], bool]:
    if not records:
        return records, False
    balanced = list(records)
    applied = False
    target_min_pct = 0.15

    while True:
        counts = pd.Series([record["regime"] for record in balanced]).value_counts().to_dict()
        total = len(balanced)
        underrepresented = [
            regime for regime in ["CRASH", "BEAR", "BULL", "SIDEWAYS"] if counts.get(regime, 0) / total < target_min_pct
        ]
        underrepresented = [regime for regime in underrepresented if any(record["regime"] == regime for record in records)]
        if not underrepresented:
            break
        for regime in underrepresented:
            candidates = [record for record in records if record["regime"] == regime]
            balanced.extend(candidates)
            applied = True
        if len(balanced) > len(records) * 10:
            break
    return balanced, applied


def save_arrow(records: list[dict], path: Path) -> None:
    Dataset = require_datasets()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    Dataset.from_list(records).save_to_disk(str(path))


def load_raw_cache(raw_path: Path, raw_fallback_path: Path) -> pd.DataFrame | None:
    if raw_path.exists():
        try:
            status(f"Using cached raw data: {raw_path}")
            return pd.read_parquet(raw_path)
        except ImportError:
            status("Parquet cache exists, but no parquet engine is installed. Trying pickle fallback...")
    if raw_fallback_path.exists():
        status(f"Using cached raw data: {raw_fallback_path}")
        return pd.read_pickle(raw_fallback_path)
    return None


def save_raw_cache(raw: pd.DataFrame, raw_path: Path, raw_fallback_path: Path) -> Path:
    try:
        raw.to_parquet(raw_path, index=False)
        return raw_path
    except ImportError:
        status("No parquet engine found. Saving raw cache as pickle instead.")
        raw.to_pickle(raw_fallback_path)
        return raw_fallback_path


def load_or_fetch_crypto_raw() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "btc_hourly_raw.parquet"
    raw_fallback_path = DATA_DIR / "btc_hourly_raw.pkl"
    cached = load_raw_cache(raw_path, raw_fallback_path)
    if cached is not None:
        return cached

    status("Fetching BTCUSDT hourly klines from Binance...")
    klines = fetch_klines()
    status("Fetching BTCUSDT funding rates from Binance Futures...")
    funding = fetch_funding_rates()
    raw = merge_funding(klines, funding)
    saved_path = save_raw_cache(raw, raw_path, raw_fallback_path)
    status(f"Saved raw merged data: {saved_path}")
    return raw


def load_or_fetch_etf_raw(symbol: str) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / f"{symbol.lower()}_daily_raw.parquet"
    raw_fallback_path = DATA_DIR / f"{symbol.lower()}_daily_raw.pkl"
    cached = load_raw_cache(raw_path, raw_fallback_path)
    if cached is not None:
        return cached

    status(f"Fetching {symbol} daily OHLCV from Yahoo Finance...")
    raw = fetch_etf_ohlcv(symbol)
    saved_path = save_raw_cache(raw, raw_path, raw_fallback_path)
    status(f"Saved raw data for {symbol}: {saved_path}")
    return raw


def prepare_crypto() -> None:
    train_path = DATA_DIR / "train.arrow"
    val_path = DATA_DIR / "val.arrow"
    test_path = DATA_DIR / "test.arrow"
    metadata_path = DATA_DIR / "metadata.json"

    raw = load_or_fetch_crypto_raw()
    df = add_regimes(add_features(raw), CRYPTO_CRASH_THRESHOLD, CRYPTO_BEAR_THRESHOLD)
    df = df.dropna(subset=FEATURE_COLUMNS + ["regime"]).reset_index(drop=True)
    print_regime_distribution(df)

    train_df = df[df["timestamp"] < TRAIN_CUTOFF]
    val_df = df[(df["timestamp"] >= TRAIN_CUTOFF) & (df["timestamp"] < VAL_CUTOFF)]
    test_df = df[df["timestamp"] >= VAL_CUTOFF]

    train_records = make_windows(train_df, WINDOW_SIZE, WINDOW_STEP)
    val_records = make_windows(val_df, WINDOW_SIZE, WINDOW_STEP)
    test_records = make_windows(test_df, WINDOW_SIZE, WINDOW_STEP)
    balanced_train_records, oversampling_applied = balance_train_windows(train_records)

    save_arrow(balanced_train_records, train_path)
    save_arrow(val_records, val_path)
    save_arrow(test_records, test_path)

    train_regime_counts = pd.Series([record["regime"] for record in balanced_train_records]).value_counts().to_dict()
    metadata = {
        "asset_class": "crypto",
        "symbols": [SYMBOL],
        "total_candles": int(len(df)),
        "train_windows": int(len(balanced_train_records)),
        "val_windows": int(len(val_records)),
        "test_windows": int(len(test_records)),
        "regime_distribution_train": {key: int(value) for key, value in train_regime_counts.items()},
        "regime_oversampling_applied": bool(oversampling_applied),
        "feature_columns": FEATURE_COLUMNS,
        "created_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("", flush=True)
    print("Saved fine-tuning data:", flush=True)
    print(f"  train : {train_path} ({len(balanced_train_records)} windows)", flush=True)
    print(f"  val   : {val_path} ({len(val_records)} windows)", flush=True)
    print(f"  test  : {test_path} ({len(test_records)} windows)", flush=True)
    print(f"  meta  : {metadata_path}", flush=True)


def prepare_etf(symbols: list[str]) -> None:
    train_path = DATA_DIR / "etf_train.arrow"
    val_path = DATA_DIR / "etf_val.arrow"
    test_path = DATA_DIR / "etf_test.arrow"
    metadata_path = DATA_DIR / "etf_metadata.json"

    all_train: list[dict] = []
    all_val: list[dict] = []
    all_test: list[dict] = []
    total_candles = 0

    for symbol in symbols:
        raw = load_or_fetch_etf_raw(symbol)
        df = add_regimes(add_features(raw), ETF_CRASH_THRESHOLD, ETF_BEAR_THRESHOLD)
        df = df.dropna(subset=FEATURE_COLUMNS + ["regime"]).reset_index(drop=True)
        total_candles += len(df)
        status(f"{symbol}: {len(df)} usable daily bars")
        print_regime_distribution(df)

        # Bucket by each window's target start time (not by slicing the
        # dataframe first) - a 380-bar window doesn't fit inside a 9-month
        # val/test period, so date-slicing first would silently yield zero
        # windows for those splits.
        train_records, val_records, test_records = make_windows_by_target_split(
            df, ETF_WINDOW_SIZE, ETF_LOOKBACK, ETF_WINDOW_STEP, TRAIN_CUTOFF, VAL_CUTOFF, symbol=symbol
        )
        all_train.extend(train_records)
        all_val.extend(val_records)
        all_test.extend(test_records)

    balanced_train_records, oversampling_applied = balance_train_windows(all_train)

    save_arrow(balanced_train_records, train_path)
    save_arrow(all_val, val_path)
    save_arrow(all_test, test_path)

    train_regime_counts = pd.Series([record["regime"] for record in balanced_train_records]).value_counts().to_dict()
    metadata = {
        "asset_class": "etf",
        "symbols": symbols,
        "total_candles": int(total_candles),
        "train_windows": int(len(balanced_train_records)),
        "val_windows": int(len(all_val)),
        "test_windows": int(len(all_test)),
        "regime_distribution_train": {key: int(value) for key, value in train_regime_counts.items()},
        "regime_oversampling_applied": bool(oversampling_applied),
        "feature_columns": FEATURE_COLUMNS,
        "created_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("", flush=True)
    print("Saved ETF fine-tuning data:", flush=True)
    print(f"  train : {train_path} ({len(balanced_train_records)} windows)", flush=True)
    print(f"  val   : {val_path} ({len(all_val)} windows)", flush=True)
    print(f"  test  : {test_path} ({len(all_test)} windows)", flush=True)
    print(f"  meta  : {metadata_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kronos fine-tuning datasets (crypto or ETF).")
    parser.add_argument("--asset-class", choices=["crypto", "etf"], default="crypto")
    parser.add_argument(
        "--symbols",
        default=",".join(ETF_FINETUNE_UNIVERSE),
        help="Comma-separated ETF tickers (only used with --asset-class etf).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.asset_class == "crypto":
        prepare_crypto()
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        prepare_etf(symbols)


if __name__ == "__main__":
    main()
