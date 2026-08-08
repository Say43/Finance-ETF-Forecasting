# Methodology

## Data

Daily OHLCV for US-listed ETFs from Yahoo Finance via `yfinance`
(split/dividend-adjusted, `auto_adjust=True`). No paid or restricted data is
used anywhere in this project. Trading-day calendars come from
`pandas-market-calendars` (NYSE schedule) so forecast timestamps skip weekends
and holidays correctly.

## Model

Base model: [`NeoQuasar/Kronos-small`](https://huggingface.co/NeoQuasar/Kronos-small)
(~25M parameters), a candlestick-sequence foundation model with a two-stage
BSQ tokenizer, fine-tuned on our ETF windows. Context length 360 daily bars,
prediction horizon 20 trading days. See [docs/reproduce.md](reproduce.md) for
the exact fine-tuning recipe.

## Calibration (`kronos_etf/calibration.py`)

Given a raw median forecast, `last_close`, and the lookback window's close
series:

1. **Bias correction** — per-horizon-step mean residual in return space,
   estimated on held-out calibration windows and subtracted from the forecast.
2. **Forecast combination** — the bias-corrected forecast is blended with a
   recent-drift term (`mu * horizon`, `mu` = mean of the last 60 simple
   returns) and an implicit naive (flat) forecast:
   `combined = w_model * model_return + w_drift * drift_return`, with
   `w_model + w_drift <= 1` (residual weight goes to the naive forecast).
   Weights are chosen by grid search minimizing pooled MAE on the calibration
   set.
3. **Split conformal intervals** — residuals from the calibration set,
   normalized by realized volatility (std of the trailing 60 daily returns,
   floored at 1e-4), give distribution-free 80%/90% quantiles `q80`, `q90`
   applied symmetrically around the combined forecast, scaled by current
   volatility.

Calibration is fit **only** on windows whose target period ends strictly before
the backtest's test region — no leakage between fitting and evaluation.

## Evaluation

- **Walk-forward backtest** (`scripts/backtest.py`): 20 non-overlapping test
  windows per symbol, most recent data last. Reports both raw and calibrated
  metrics: directional accuracy, MAE/MAPE vs the naive (last-value) baseline,
  empirical coverage of the 80%/90% bands, and mean band width.
- **Diebold–Mariano test**: per-window loss differential (calibrated MAE minus
  naive MAE), Newey–West HAC variance with `lag = horizon - 1` and Bartlett
  weights, two-sided p-value via the normal CDF. Implemented from scratch
  (`math.erf`) — no SciPy dependency for this specific test.
- **Volatility scoring**: Pearson/Spearman correlation with realized
  volatility, RMSE, and QLIKE (`E[realized²/pred² − ln(realized²/pred²) − 1]`),
  the proper scoring rule for volatility forecasts used in the GARCH
  literature.

## Portfolio backtests (`research/`)

Monthly-rebalanced, equal-weight sleeves unless noted; transaction costs 0.10%
of turnover; positions decided using only data available at time `t` and
applied to the return from `t` to `t+1` (no lookahead). Two independent
building blocks were tested:

- **Cross-sectional momentum**: rank a universe of ETFs by trailing 12-1-month
  return, hold the top N equally weighted.
- **Time-series trend + volatility targeting**: per-ETF, long/flat by the sign
  of the trailing-window median monthly log return; position size scaled to a
  target annualized volatility using a trailing volatility estimate (realized,
  EWMA, or GARCH(1,1) via the `arch` package), capped at no leverage.

Statistics reported: CAGR, annualized volatility, Sharpe ratio, and maximum
drawdown, computed from monthly returns over the stated common window.

## What is deliberately *not* done

- No leveraged or inverse products, no options, no intraday data.
- No hyperparameter sweep beyond what is reported (few, theory-motivated
  configurations rather than an exhaustive search — see the leakage/overfitting
  discussion in `NOTICE.md` and `docs/results.md`).
- No live trading integration. This is a research and evaluation codebase.
