"""Statistical post-processing for Kronos point forecasts.

Calibration parameters are intentionally model-agnostic and JSON serializable.
All returns in this module are simple returns relative to the last observed
close.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_MIN_VOL = 1e-4
_MIN_VOL_BARS = 20
_VOL_WINDOW = 60


@dataclass(frozen=True)
class CalibrationParams:
    """Parameters fitted on pre-test walk-forward forecast residuals."""

    asset_class: str
    checkpoint: str
    pred_len: int
    n_points: int
    bias: list[float]
    w_model: float
    w_drift: float
    q80: list[float]
    q90: list[float]

    def __post_init__(self) -> None:
        if not self.asset_class:
            raise ValueError("asset_class must not be empty")
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if isinstance(self.pred_len, bool) or self.pred_len <= 0:
            raise ValueError("pred_len must be a positive integer")
        if isinstance(self.n_points, bool) or self.n_points < 0:
            raise ValueError("n_points must be a non-negative integer")

        for name in ("bias", "q80", "q90"):
            values = getattr(self, name)
            if len(values) != self.pred_len:
                raise ValueError(
                    f"{name} must contain pred_len={self.pred_len} values, "
                    f"got {len(values)}"
                )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain only finite values")

        if any(float(value) < 0.0 for value in (*self.q80, *self.q90)):
            raise ValueError("conformal quantiles must be non-negative")
        if any(float(q90) < float(q80) for q80, q90 in zip(self.q80, self.q90)):
            raise ValueError("q90 must be greater than or equal to q80")

        weights = (float(self.w_model), float(self.w_drift))
        if not all(math.isfinite(weight) and weight >= 0.0 for weight in weights):
            raise ValueError("calibration weights must be finite and non-negative")
        if sum(weights) > 1.0 + 1e-12:
            raise ValueError("w_model + w_drift must not exceed 1")


def load_calibration(path: str | Path) -> CalibrationParams | None:
    """Load calibration parameters, returning ``None`` when *path* is absent."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        return None

    with calibration_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    payload.pop("created_utc", None)
    return CalibrationParams(**payload)


def save_calibration(params: CalibrationParams, path: str | Path) -> None:
    """Persist calibration parameters and their creation timestamp as JSON."""
    calibration_path = Path(path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(params)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _validated_close_series(x_close: pd.Series) -> pd.Series:
    close = pd.Series(x_close, copy=False).astype(float)
    if len(close) < _MIN_VOL_BARS:
        raise ValueError(
            f"x_close must contain at least {_MIN_VOL_BARS} bars, got {len(close)}"
        )
    values = close.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("x_close must contain only finite values")
    if (values <= 0.0).any():
        raise ValueError("x_close values must be positive")
    return close


def _recent_simple_returns(x_close: pd.Series) -> pd.Series:
    close = _validated_close_series(x_close).tail(_VOL_WINDOW)
    return close.pct_change(fill_method=None).dropna()


def realized_daily_vol(x_close: pd.Series) -> float:
    """Return sample volatility of recent simple returns, floored at 1e-4."""
    returns = _recent_simple_returns(x_close)
    if returns.empty:
        return _MIN_VOL
    volatility = float(returns.std(ddof=1))
    if not math.isfinite(volatility):
        return _MIN_VOL
    return max(volatility, _MIN_VOL)


def apply_calibration(
    pred_close: pd.Series | np.ndarray | list[float],
    last_close: float,
    x_close: pd.Series,
    params: CalibrationParams,
) -> dict[str, pd.Series]:
    """Bias-correct, combine and add volatility-scaled conformal intervals."""
    last_close = float(last_close)
    if not math.isfinite(last_close) or last_close <= 0.0:
        raise ValueError("last_close must be finite and positive")

    if isinstance(pred_close, pd.Series):
        index = pred_close.index
        forecast = pred_close.to_numpy(dtype=float)
    else:
        forecast = np.asarray(pred_close, dtype=float)
        index = pd.RangeIndex(len(forecast)) if forecast.ndim == 1 else None

    if forecast.ndim != 1:
        raise ValueError("pred_close must be one-dimensional")
    if len(forecast) != params.pred_len:
        raise ValueError(
            f"pred_close must contain pred_len={params.pred_len} values, "
            f"got {len(forecast)}"
        )
    if not np.isfinite(forecast).all():
        raise ValueError("pred_close must contain only finite values")

    returns = _recent_simple_returns(x_close)
    mu = float(returns.mean()) if not returns.empty else 0.0
    sigma = realized_daily_vol(x_close)

    horizons = np.arange(1, params.pred_len + 1, dtype=float)
    model_return = forecast / last_close - 1.0 - np.asarray(params.bias, dtype=float)
    drift_return = mu * horizons
    combined_return = (
        float(params.w_model) * model_return
        + float(params.w_drift) * drift_return
    )

    center = last_close * (1.0 + combined_return)
    q80 = np.asarray(params.q80, dtype=float)
    q90 = np.asarray(params.q90, dtype=float)

    def series(values: np.ndarray) -> pd.Series:
        return pd.Series(values, index=index, dtype=float)

    return {
        "close": series(center),
        "lo80": series(last_close * (1.0 + combined_return - q80 * sigma)),
        "hi80": series(last_close * (1.0 + combined_return + q80 * sigma)),
        "lo90": series(last_close * (1.0 + combined_return - q90 * sigma)),
        "hi90": series(last_close * (1.0 + combined_return + q90 * sigma)),
    }
