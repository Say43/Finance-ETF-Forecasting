from __future__ import annotations

import sys
import time
from inspect import signature
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_candidates() -> list[Path]:
    """Where to look for the upstream Kronos model code (MIT, see NOTICE.md).

    Set KRONOS_REPO to point at your clone; otherwise a few conventional
    locations are tried. The upstream repo is not vendored here so that its
    code stays under its own license and version control.
    """
    import os

    env = os.environ.get("KRONOS_REPO")
    candidates = [Path(env)] if env else []
    candidates += [
        REPO_ROOT / "external" / "Kronos",
        REPO_ROOT.parent / "Kronos",
        Path.home() / "Kronos",
    ]
    return candidates


def resolve_kronos_repo() -> Path:
    for candidate in _repo_candidates():
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Kronos model code not found. Clone https://github.com/shiyu-coder/Kronos "
        "and set KRONOS_REPO to its path, or place it at one of: "
        + ", ".join(str(path) for path in _repo_candidates())
    )


KRONOS_REPO_PATH = resolve_kronos_repo()
if str(KRONOS_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(KRONOS_REPO_PATH))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402


def detect_device() -> tuple[str, str]:
    """Detect best available compute device. Returns (device_str, info_label)."""
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return "cuda", f"CUDA | {name} | {vram_gb:.1f} GB VRAM"
    import os
    import platform

    return "cpu", f"CPU | {platform.processor() or platform.machine()} | {os.cpu_count()} cores"


DEVICE_STR, DEVICE_INFO = detect_device()
BATCH_MODE = {"mode": "unknown"}
APP_ROOT = REPO_ROOT
DEFAULT_FINETUNED_CHECKPOINT = Path("finetune/checkpoints/best_clean")
DEFAULT_KRONOS_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
PRETRAINED_MODEL_LABEL = f"BASE MODEL: {DEFAULT_KRONOS_MODEL}"
MODEL_VERSION = {"label": PRETRAINED_MODEL_LABEL}


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
KLINE_FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [*KLINE_FEATURE_COLUMNS, "funding_rate"]


def status(message: str) -> None:
    print(f"[STATUS] {message}", flush=True)


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    if limit <= 0 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000 for the Binance klines endpoint.")

    try:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Binance klines request failed: {exc}") from exc
    if not response.ok:
        raise RuntimeError(f"Binance klines request failed with HTTP {response.status_code}: {response.text}")
    data = response.json()
    if not isinstance(data, list) or not data:
        raise ValueError("Binance returned no kline data.")

    df = pd.DataFrame(data, columns=BINANCE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df = df[["timestamp", *KLINE_FEATURE_COLUMNS]]
    df[KLINE_FEATURE_COLUMNS] = df[KLINE_FEATURE_COLUMNS].astype(float)
    return df.reset_index(drop=True)


def fetch_etf_klines(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch free daily OHLCV data for an ETF/equity ticker via Yahoo Finance."""
    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    df[KLINE_FEATURE_COLUMNS] = df[KLINE_FEATURE_COLUMNS].astype(float)
    return df[["timestamp", *KLINE_FEATURE_COLUMNS]].reset_index(drop=True)


def fetch_funding_rate(symbol: str = "BTCUSDT", limit: int = 400) -> pd.DataFrame:
    """Fetch Binance Futures funding rates for a symbol."""
    try:
        response = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": limit},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Binance funding rate request failed: {exc}") from exc
    if not response.ok:
        raise RuntimeError(
            f"Binance funding rate request failed with HTTP {response.status_code}: {response.text}"
        )

    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("Binance returned invalid funding rate data.")

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert(None)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)


def merge_funding_rate(df: pd.DataFrame, fr_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent funding rate at or before each market timestamp."""
    enriched = df.copy()
    funding = fr_df.copy()
    if "funding_rate" in enriched.columns:
        enriched = enriched.drop(columns=["funding_rate"])

    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True).dt.tz_convert(None)
    funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True).dt.tz_convert(None)

    enriched = enriched.sort_values("timestamp").reset_index(drop=True)
    funding = funding.sort_values("timestamp").reset_index(drop=True)

    if funding.empty:
        enriched["funding_rate"] = 0.0
        return enriched

    enriched = pd.merge_asof(enriched, funding, on="timestamp", direction="backward")
    enriched["funding_rate"] = enriched["funding_rate"].fillna(0.0).astype(float)
    return enriched


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = APP_ROOT / path
    return path


def load_kronos_predictor(
    model_name: str = DEFAULT_KRONOS_MODEL,
    max_context: int = 512,
    use_finetuned: bool = False,
    checkpoint_path: str | Path = DEFAULT_FINETUNED_CHECKPOINT,
):
    if use_finetuned:
        return load_finetuned_predictor(checkpoint_path=checkpoint_path)

    if DEVICE_STR == "cuda":
        torch.cuda.empty_cache()

    if "amazon" in str(model_name).lower() or "chronos-t5" in str(model_name).lower():
        raise RuntimeError(f"Amazon Chronos model is not allowed: {model_name}")

    MODEL_VERSION["label"] = PRETRAINED_MODEL_LABEL
    status(f"BASE MODEL: {model_name}")
    tokenizer = KronosTokenizer.from_pretrained(DEFAULT_KRONOS_TOKENIZER)
    model = Kronos.from_pretrained(model_name)
    assert "amazon" not in str(model_name).lower(), "Wrong model loaded - expected GitHub Kronos, got Amazon Chronos"
    if hasattr(model, "to"):
        model.to(DEVICE_STR)
    predictor = KronosPredictor(model, tokenizer, max_context=max_context)
    if hasattr(predictor, "model"):
        predictor.model.to(DEVICE_STR)
    elif hasattr(predictor, "to"):
        predictor.to(DEVICE_STR)
    status(f"Compute Device: {DEVICE_INFO}")
    return predictor


def load_finetuned_predictor(
    checkpoint_path: str | Path = DEFAULT_FINETUNED_CHECKPOINT,
) -> KronosPredictor:
    """Load fine-tuned GitHub Kronos predictor checkpoint."""
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {resolved_checkpoint}")

    relative_label = str(Path(checkpoint_path)).replace("\\", "/")
    MODEL_VERSION["label"] = f"FINE-TUNED: {relative_label} (base: Kronos-small)"
    status(f"FINE-TUNED: {relative_label} (base: Kronos-small)")
    tokenizer = KronosTokenizer.from_pretrained(DEFAULT_KRONOS_TOKENIZER)
    model = Kronos.from_pretrained(str(resolved_checkpoint))
    assert "kronos" in model.__class__.__name__.lower(), "Wrong model loaded - expected GitHub Kronos"
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    if hasattr(predictor, "model"):
        predictor.model.to(DEVICE_STR)
    status(f"Compute Device: {DEVICE_INFO}")
    return predictor


def make_future_timestamps(
    last_timestamp: pd.Timestamp,
    pred_len: int,
    step: timedelta = timedelta(hours=1),
    calendar: str | None = None,
) -> pd.Series:
    """Generate future bar timestamps. For 24/7 assets (crypto) this is a fixed
    step; for exchange-listed assets (calendar given, e.g. "XNYS") it skips
    weekends/holidays using that exchange's real trading schedule."""
    if calendar is None:
        return pd.Series([last_timestamp + step * (i + 1) for i in range(pred_len)])

    import pandas_market_calendars as mcal

    schedule_days = mcal.get_calendar(calendar).schedule(
        start_date=last_timestamp.date(),
        end_date=last_timestamp + timedelta(days=int(pred_len * 2.5) + 10),
    )
    future_days = schedule_days.index[schedule_days.index > pd.Timestamp(last_timestamp.date())]
    if len(future_days) < pred_len:
        raise RuntimeError(
            f"Trading calendar {calendar!r} did not yield {pred_len} future sessions "
            f"after {last_timestamp}; only found {len(future_days)}."
        )
    return pd.Series(future_days[:pred_len]).reset_index(drop=True)


def score_prediction(history_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    hist_close = history_df["close"].astype(float).reset_index(drop=True)
    pred_close = pred_df["close"].astype(float).reset_index(drop=True)

    last_close = float(hist_close.iloc[-1])
    first_close = float(pred_close.iloc[0])
    jump_penalty = abs(first_close - last_close) / max(last_close, 1.0)

    hist_returns = hist_close.pct_change().dropna()
    pred_returns = pred_close.pct_change().dropna()

    hist_vol = float(hist_returns.std()) if not hist_returns.empty else 0.0
    pred_vol = float(pred_returns.std()) if not pred_returns.empty else 0.0
    vol_penalty = abs(pred_vol - hist_vol)

    hist_trend = float(hist_close.iloc[-1] - hist_close.iloc[-24]) if len(hist_close) >= 24 else 0.0
    pred_trend = float(pred_close.iloc[-1] - pred_close.iloc[0])
    trend_penalty = abs(pred_trend - hist_trend) / max(abs(last_close), 1.0)

    max_drawdown = float((pred_close.cummax() - pred_close).max()) / max(last_close, 1.0)
    tail_floor = float(hist_close.tail(24).min())
    tail_floor_penalty = max(0.0, (tail_floor - float(pred_close.min())) / max(last_close, 1.0))
    end_drop_penalty = max(0.0, (float(pred_close.iloc[0]) - float(pred_close.iloc[-1])) / max(last_close, 1.0))
    cliff_penalty = 0.0
    if not pred_returns.empty:
        cliff_penalty = max(0.0, -float(pred_returns.min()) - max(hist_vol * 2.5, 0.01))

    return (
        jump_penalty * 5.0
        + vol_penalty * 10.0
        + trend_penalty * 3.0
        + max_drawdown * 8.0
        + tail_floor_penalty * 12.0
        + end_drop_penalty * 10.0
        + cliff_penalty * 18.0
    )


def align_prediction_to_history(history_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    aligned = pred_df.copy()
    price_cols = ["open", "high", "low", "close"]
    aligned[price_cols] = aligned[price_cols].astype(float)
    close_shift = float(history_df["close"].iloc[-1] - aligned["close"].iloc[0])
    aligned[price_cols] = aligned[price_cols] + close_shift

    blend = min(4, len(aligned))
    if blend > 0:
        start = history_df["close"].iloc[-1]
        target = aligned["close"].iloc[:blend].to_numpy()
        weights = [0.85, 0.55, 0.3, 0.15][:blend]
        for i, weight in enumerate(weights):
            aligned.iloc[i, aligned.columns.get_loc("close")] = start * weight + target[i] * (1 - weight)

    aligned["volume"] = aligned["volume"].clip(lower=0)
    if "amount" in aligned.columns:
        aligned["amount"] = aligned["amount"].clip(lower=0)
    return aligned


def smooth_prediction(history_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    smoothed = pred_df.copy()
    prev_close = float(history_df["close"].iloc[-1])
    raw_close = smoothed["close"].astype(float).to_numpy()
    adjusted = []

    # Volatility-adaptive step clip: a fixed 1.2%/1.8% cap only makes sense for
    # BTC's hourly volatility. Deriving it from the history's own realized
    # per-bar volatility lets the same function work for ETF daily bars too.
    hist_returns = history_df["close"].astype(float).pct_change().dropna()
    realized_vol = float(hist_returns.std()) if not hist_returns.empty else 0.012
    base_step = max(realized_vol * 2.5, 0.003)

    for i, value in enumerate(raw_close):
        max_step = base_step if i < 6 else base_step * 1.5
        upper = prev_close * (1 + max_step)
        lower = prev_close * (1 - max_step)
        clipped = min(max(value, lower), upper)
        blended = prev_close * 0.35 + clipped * 0.65
        adjusted.append(blended)
        prev_close = blended

    smoothed["close"] = adjusted

    close_delta = smoothed["close"] - pred_df["close"].astype(float)
    for col in ["open", "high", "low"]:
        smoothed[col] = pred_df[col].astype(float) + close_delta
    smoothed["high"] = smoothed[["open", "close", "high"]].max(axis=1)
    smoothed["low"] = smoothed[["open", "close", "low"]].min(axis=1)
    if "volume" in smoothed.columns:
        smoothed["volume"] = smoothed["volume"].astype(float).rolling(3, min_periods=1).mean()
    if "amount" in smoothed.columns:
        smoothed["amount"] = smoothed["amount"].astype(float).rolling(3, min_periods=1).mean()
    return smoothed


def build_candidates() -> list[dict[str, float | int]]:
    return [
        {"T": 0.9, "top_k": 0, "top_p": 0.95, "sample_count": 10},
        {"T": 0.85, "top_k": 5, "top_p": 1.0, "sample_count": 15},
        {"T": 0.8, "top_k": 0, "top_p": 0.90, "sample_count": 20},
        {"T": 0.75, "top_k": 10, "top_p": 1.0, "sample_count": 20},
    ]


def _signal_sharpness_score(samples: list[pd.DataFrame], last_close: float) -> float:
    """
    Signal Sharpness Score: measures how concentrated and directional
    the forecast is relative to the uncertainty band.
    """
    if not samples:
        return 0.0
    sample_means = [float(sample["close"].mean()) for sample in samples]
    import numpy as np

    arr = np.array(sample_means)
    band_std = arr.std()
    if band_std == 0:
        return 0.0
    return float(abs(arr.mean() - last_close) / band_std)


def _filter_call_kwargs(fn, kwargs: dict) -> dict:
    params = signature(fn).parameters
    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _sampling_kwargs_for(fn, cfg: dict[str, float | int]) -> dict:
    params = signature(fn).parameters
    kwargs = {}
    effective_top_p = 1.0 if int(cfg["top_k"]) > 0 and float(cfg["top_p"]) < 1.0 else float(cfg["top_p"])
    if "T" in params:
        kwargs["T"] = float(cfg["T"])
    elif "temperature" in params:
        kwargs["temperature"] = float(cfg["T"])

    if "top_k" in params:
        kwargs["top_k"] = int(cfg["top_k"])
    elif "topk" in params:
        kwargs["topk"] = int(cfg["top_k"])

    if "top_p" in params:
        kwargs["top_p"] = effective_top_p
    elif "topp" in params:
        kwargs["topp"] = effective_top_p

    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
        kwargs.setdefault("T", float(cfg["T"]))
        kwargs.setdefault("top_k", int(cfg["top_k"]))
        kwargs.setdefault("top_p", effective_top_p)
    return kwargs


def _sample_to_dataframe(sample, x_df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
    if isinstance(sample, pd.DataFrame):
        return sample.copy()

    if hasattr(sample, "detach"):
        values = sample.detach().cpu().numpy()
    else:
        values = sample
    values = pd.Series(values).astype(float).to_numpy().reshape(-1)[:pred_len]
    out = pd.DataFrame({"close": values})
    for col in x_df.columns:
        if col == "close":
            continue
        if col in {"open", "high", "low"}:
            out[col] = out["close"]
        else:
            out[col] = float(x_df[col].iloc[-1])
    return out[x_df.columns]


def _call_predict(
    predictor: KronosPredictor,
    x_df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int,
    cfg: dict[str, float | int],
) -> pd.DataFrame:
    kwargs = {
        "df": x_df,
        "x_timestamp": x_timestamp,
        "y_timestamp": y_timestamp,
        "pred_len": pred_len,
        "sample_count": 1,
        "verbose": False,
        **_sampling_kwargs_for(predictor.predict, cfg),
    }
    return predictor.predict(**_filter_call_kwargs(predictor.predict, kwargs))


def _call_predict_batch(
    predictor: KronosPredictor,
    x_df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int,
    cfg: dict[str, float | int],
) -> list[pd.DataFrame] | None:
    if not hasattr(predictor, "predict_batch"):
        return None

    input_list = [x_df.copy() for _ in range(int(cfg["sample_count"]))]
    x_timestamp_list = [x_timestamp.copy() for _ in input_list]
    y_timestamp_list = [y_timestamp.copy() for _ in input_list]
    common_kwargs = {
        "df_list": input_list,
        "dfs": input_list,
        "input_list": input_list,
        "x_timestamp_list": x_timestamp_list,
        "x_timestamps": x_timestamp_list,
        "x_timestamp": x_timestamp_list,
        "y_timestamp_list": y_timestamp_list,
        "y_timestamps": y_timestamp_list,
        "y_timestamp": y_timestamp_list,
        "pred_len": pred_len,
        "sample_count": 1,
        "verbose": False,
        **_sampling_kwargs_for(predictor.predict_batch, cfg),
    }
    fn = predictor.predict_batch
    attempts = [
        lambda: fn(**_filter_call_kwargs(fn, common_kwargs)),
        lambda: fn(input_list, x_timestamp_list, y_timestamp_list, pred_len=pred_len, **_filter_call_kwargs(fn, {**_sampling_kwargs_for(fn, cfg), "sample_count": 1, "verbose": False})),
        lambda: fn(input_list, pred_len=pred_len, **_filter_call_kwargs(fn, {**_sampling_kwargs_for(fn, cfg), "sample_count": 1, "verbose": False})),
    ]
    for attempt in attempts:
        try:
            raw = attempt()
            if raw is None:
                continue
            if isinstance(raw, pd.DataFrame):
                raw_samples = [raw]
            elif hasattr(raw, "detach"):
                arr = raw.detach().cpu()
                raw_samples = [arr[i] for i in range(arr.shape[0])]
            else:
                raw_samples = list(raw)
            return [_sample_to_dataframe(sample, x_df, pred_len) for sample in raw_samples]
        except TypeError:
            continue
    return None


def _predict_samples(
    predictor: KronosPredictor,
    x_df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int,
    cfg: dict[str, float | int],
    log=status,
) -> list[pd.DataFrame]:
    batch_samples = _call_predict_batch(predictor, x_df, x_timestamp, y_timestamp, pred_len, cfg)
    if batch_samples is not None:
        BATCH_MODE["mode"] = "predict_batch"
        samples = batch_samples
    else:
        if BATCH_MODE["mode"] != "sequential" and log is not None:
            log("[WARNUNG] predict_batch nicht verfügbar, nutze sequenziellen Modus")
        BATCH_MODE["mode"] = "sequential"
        samples = []
        for _ in range(int(cfg["sample_count"])):
            samples.append(_call_predict(predictor, x_df, x_timestamp, y_timestamp, pred_len, cfg))

    processed = []
    for sample_df in samples:
        sample_df = align_prediction_to_history(x_df, sample_df)
        sample_df = smooth_prediction(x_df, sample_df)
        processed.append(sample_df.copy())
    return processed


def choose_best_prediction(
    predictor: KronosPredictor,
    x_df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int,
    log= status,
) -> tuple[pd.DataFrame, dict[str, float | int], float, list[pd.DataFrame], list[float]]:
    BATCH_MODE["mode"] = "unknown"
    band_cfg = {"T": 0.9, "top_k": 0, "top_p": 0.92, "sample_count": 24}
    last_close = float(x_df["close"].iloc[-1])
    all_scores: list[float] = []

    band_start = time.perf_counter()
    all_predictions = _predict_samples(predictor, x_df, x_timestamp, y_timestamp, pred_len, band_cfg, log=log)
    band_elapsed = time.perf_counter() - band_start
    if log is not None:
        log(
            f"Erzeuge Forecast-Band: T={band_cfg['T']}, top_k={band_cfg['top_k']}, "
            f"top_p={band_cfg['top_p']}, samples={band_cfg['sample_count']} | "
            f"batch_mode={BATCH_MODE['mode'] == 'predict_batch'} | {band_elapsed:.1f}s"
        )
    if not all_predictions:
        raise RuntimeError("No valid forecast band samples were produced.")

    median_forecast_df = pd.DataFrame(
        {
            col: pd.concat([sample[col] for sample in all_predictions], axis=1).median(axis=1)
            for col in all_predictions[0].columns
        }
    )

    best_cfg = band_cfg
    best_score = _signal_sharpness_score(all_predictions, last_close)
    return median_forecast_df, best_cfg, best_score, all_predictions, all_scores


def naive_baseline(last_close: float, y_timestamp: Iterable[pd.Timestamp]) -> pd.DataFrame:
    index = pd.Index(list(y_timestamp), name="timestamp")
    return pd.DataFrame({"close": [last_close] * len(index)}, index=index)
