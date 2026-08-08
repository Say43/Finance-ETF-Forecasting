# Research scripts

Standalone, self-contained backtests that don't belong in the main package —
each downloads its own data and prints its own report. Findings are summarized
in [../docs/results.md](../docs/results.md); this is the reproducible source.

**Kronos-dependent** (need a fine-tuned checkpoint + calibration file, see
[../docs/reproduce.md](../docs/reproduce.md)):

- `simulate_month.py` — walk-forward daily paper-trade of the calibrated
  signal vs buy & hold, with realistic hysteresis and costs.
- `resim.py` — offline re-simulation from logged signals: zero-fee edge check
  and a sell-threshold grid search.
- `bear_test.py` — single-decision test at the onset of a broad-market
  selloff.
- `vol_race.py` — Kronos's volatility forecast vs GARCH(1,1)/EWMA/persistence,
  scored by correlation, RMSE and QLIKE.
- `vol_ensemble.py` — does combining Kronos with GARCH improve on either alone?

**Model-free portfolio construction** (no GPU or checkpoint needed):

- `xsec_momentum.py`, `xsec_momentum_v2.py` — cross-sectional momentum over a
  broad ETF universe vs 60/40 and buy & hold, with a fair-window/robustness
  check and volatility targeting.
- `combo_test.py` — blending a momentum sleeve with 60/40; the one setup that
  robustly beat 60/40 risk-adjusted.
- `dynamic_tilt.py` — does dynamically timing the momentum/60-40 mix (trend,
  vol regime) beat the static blend? (It does not.)
- `slope_signal.py` — direction from the trailing-window median slope alone
  (time-series momentum).
- `slope_plus_vol.py` — the strongest configuration found: trailing-slope
  direction combined with volatility-targeted position sizing.
- `single_stocks.py` — does the slope+vol architecture do more for single
  stocks (higher crash risk) than for ETFs? (It does not — overnight gap risk
  isn't hedgeable by a trend/vol system.)

Run any script directly, e.g.:

```bash
python research/slope_plus_vol.py
```
