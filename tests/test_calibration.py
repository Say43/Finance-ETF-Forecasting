from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kronos_etf.calibration import (
    CalibrationParams,
    apply_calibration,
    load_calibration,
    realized_daily_vol,
    save_calibration,
)


def make_params(**overrides) -> CalibrationParams:
    values = {
        "asset_class": "etf",
        "checkpoint": "finetune/checkpoints_etf/best_clean",
        "pred_len": 2,
        "n_points": 105,
        "bias": [0.01, 0.02],
        "w_model": 0.5,
        "w_drift": 0.25,
        "q80": [2.0, 2.0],
        "q90": [3.0, 3.0],
    }
    values.update(overrides)
    return CalibrationParams(**values)


def test_save_load_round_trip_and_missing_file(tmp_path) -> None:
    path = tmp_path / "nested" / "calibration.json"
    params = make_params()

    assert load_calibration(path) is None
    save_calibration(params, path)

    assert load_calibration(path) == params
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["created_utc"].endswith("+00:00")
    assert payload["n_points"] == 105


def test_realized_daily_vol_uses_only_latest_60_bars() -> None:
    old_segment = 100.0 * np.cumprod(np.resize([1.2, 0.8], 40))
    stable_start = old_segment[-1]
    recent_segment = stable_start * np.power(1.01, np.arange(1, 61))
    close = pd.Series(np.concatenate([old_segment, recent_segment]))

    assert realized_daily_vol(close) == pytest.approx(1e-4)


def test_realized_daily_vol_matches_sample_std_and_applies_floor() -> None:
    returns = np.resize([0.01, -0.02, 0.03, 0.0], 59)
    close = pd.Series(100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns])))

    assert realized_daily_vol(close) == pytest.approx(
        pd.Series(returns).std(ddof=1)
    )
    assert realized_daily_vol(pd.Series([100.0] * 20)) == pytest.approx(1e-4)


def test_realized_daily_vol_rejects_bad_history() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        realized_daily_vol(pd.Series([100.0] * 19))
    with pytest.raises(ValueError, match="finite"):
        realized_daily_vol(pd.Series([100.0] * 19 + [np.nan]))
    with pytest.raises(ValueError, match="positive"):
        realized_daily_vol(pd.Series([100.0] * 19 + [0.0]))


def test_apply_calibration_combines_bias_model_drift_and_naive_weights() -> None:
    # Constant 1% returns imply mu=1% and sigma is floored to 0.01%.
    history = pd.Series(100.0 * np.power(1.01, np.arange(61)))
    index = pd.date_range("2026-01-02", periods=2, freq="B")
    forecast = pd.Series([110.0, 120.0], index=index)

    result = apply_calibration(forecast, 100.0, history, make_params())

    expected_center = pd.Series([104.75, 109.5], index=index)
    pd.testing.assert_series_equal(result["close"], expected_center)
    pd.testing.assert_series_equal(
        result["lo80"], expected_center - 0.02, check_names=False
    )
    pd.testing.assert_series_equal(
        result["hi80"], expected_center + 0.02, check_names=False
    )
    pd.testing.assert_series_equal(
        result["lo90"], expected_center - 0.03, check_names=False
    )
    pd.testing.assert_series_equal(
        result["hi90"], expected_center + 0.03, check_names=False
    )


def test_apply_calibration_supports_implicit_naive_only() -> None:
    params = make_params(w_model=0.0, w_drift=0.0)
    result = apply_calibration(
        [90.0, 120.0],
        100.0,
        pd.Series([100.0] * 20),
        params,
    )

    np.testing.assert_allclose(result["close"], [100.0, 100.0])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bias": [0.1]}, "bias must contain"),
        ({"q80": [2.0, -1.0]}, "non-negative"),
        ({"q90": [1.0, 3.0]}, "q90 must be"),
        ({"w_model": 0.8, "w_drift": 0.3}, "must not exceed"),
    ],
)
def test_params_reject_invalid_values(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_params(**overrides)


def test_apply_calibration_rejects_invalid_forecast() -> None:
    history = pd.Series([100.0] * 20)
    params = make_params()

    with pytest.raises(ValueError, match="pred_len"):
        apply_calibration([101.0], 100.0, history, params)
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_calibration(np.ones((2, 1)), 100.0, history, params)
    with pytest.raises(ValueError, match="last_close"):
        apply_calibration([101.0, 102.0], 0.0, history, params)
