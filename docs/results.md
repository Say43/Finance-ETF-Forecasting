# Results

All numbers below are produced by the scripts in this repository on free Yahoo
Finance daily data. Read [DISCLAIMER.md](../DISCLAIMER.md) first: the headline
result is a negative one, and none of this is investment advice.

Conventions: horizon = 20 trading days (~1 month); context = 360 daily bars;
"naive baseline" = last-value / random-walk forecast; costs, where charged, are
0.10% per unit of turnover.

---

## 1. Fine-tuning the foundation model

`NeoQuasar/Kronos-small` was fine-tuned on 22 US ETFs (broad index, sector
SPDRs, international, bonds, commodities), 15 years of daily bars, windowed into
380-bar samples split by **target** date (train < 2025-01-01, val < 2025-10-01).
**VTI, XLF and VNQ were excluded from training** and kept as an out-of-sample
holdout.

| | v1 | v2 |
|---|---|---|
| Universe | 7 ETFs | 22 ETFs |
| Learning rate | 1e-4 | 3e-5, cosine decay |
| Early stopping | none (5 epochs) | step-level, patience 4 |
| Extras | — | EMA weights, AMP fp16, wall-clock guard |
| Train windows | 5,920 | 22,637 |
| Best val loss | epoch 1 of 5 (overfit after) | 2.768 @ step 2100 (~3 epochs) |

v1 overfitted after a single epoch; v2's lower LR and early stopping fixed that.
Training ran ~29 min on a Kaggle T4.

## 2. Calibration layer

Raw model output is post-processed with three standard, leakage-free steps,
fitted on walk-forward windows strictly *before* the test region
(`scripts/calibrate.py`, 105 pooled calibration points):

1. per-horizon **bias correction** in return space,
2. **forecast combination** (Bates–Granger style shrinkage toward the naive
   forecast plus a drift term; weights by grid search),
3. **split conformal** 80% / 90% prediction intervals, scaled by realized
   volatility.

Fitted v2 parameters: `w_model = 0.45`, `w_drift = 0.10`, implied
`w_naive = 0.45`, mean bias −0.24%.

## 3. Forecast accuracy — the headline negative result

Walk-forward backtest, 20 windows per symbol (`scripts/backtest.py`).
`MAE impr.` = improvement in mean absolute error vs the naive baseline
(positive = better than naive).

| Symbol | v1 raw | v2 raw | **v2 calibrated** | cov80 | cov90 | DM p |
|---|---:|---:|---:|---:|---:|---:|
| SPY | −7.7% | −0.2% | +1.96% | 79.2% | 85.8% | 0.350 |
| QQQ | −14.7% | −4.7% | −0.83% | 73.8% | 83.5% | 0.609 |
| IWM | +2.7% | +0.9% | +3.27% | 72.7% | 84.5% | 0.071 |
| DIA | −19.5% | −8.9% | −1.17% | 78.0% | 86.2% | 0.746 |
| EFA | −20.0% | −0.1% | +2.10% | 75.2% | 84.8% | 0.214 |
| GLD | +5.7% | +9.6% | +9.02% | 84.0% | 94.2% | **0.000** |
| TLT | −6.2% | −26.1% | −7.62% | 86.2% | 94.5% | 0.083 |
| **mean (trained)** | **−8.5%** | **−4.2%** | **+0.96%** | 78.4% | 87.6% | |
| VTI *(holdout)* | | −12.6% | −3.39% | 77.2% | 84.8% | 0.261 |
| XLF *(holdout)* | | −19.7% | −8.04% | 82.8% | 92.2% | 0.002 |
| VNQ *(holdout)* | | −10.4% | −2.99% | 80.5% | 88.2% | 0.387 |

**What worked:** calibration lifted the point forecast from clearly worse than
naive to roughly naive-level, and the conformal bands are **well calibrated**
(80% band covers 78.4%, 90% covers 87.6%). Fine-tuning also generalises: on
holdouts the zero-shot base model scored −28% to −89% MAE improvement versus
−3% to −8% fine-tuned.

**What did not work:** only GLD beats the naive baseline significantly
(Diebold–Mariano p < 0.001); XLF is significantly *worse*. Everything else is
statistically indistinguishable from a random walk.

## 4. Is there a tradable edge? No.

Three independent tests, all negative.

**4.1 One month of paper trading** (`research/simulate_month.py`, daily
rebalance, €10,000, 0.10% costs, 2026-06-23 → 2026-07-23):

| | end value | return |
|---|---:|---:|
| Strategy (net) | €9,863 | −1.37% |
| Equal-weight buy & hold | €9,892 | −1.08% |
| SPY buy & hold | €10,063 | +0.63% |

The "sell on expected loss" rule never fired — the strategy was fully invested
21/21 days, so it reduced to buy-and-hold plus trading costs.

**4.2 Zero fees** (`research/resim.py` and a zero-fee sweep over all 200
backtest windows). Removing costs entirely does **not** create an edge:

| Zero-fee strategy | mean per 20d | compounded |
|---|---:|---:|
| Always long (buy & hold) | **1.32%** | **998%** |
| Long/flat on calibrated signal | 1.03% | 563% |
| Long/short on calibrated signal | 0.74% | 243% |

Directional accuracy was 59.0% against a **65.0%** market up-rate — i.e. 6
points *worse* than a clock that always says "up".

**4.3 Asymmetric thresholds** (only exit when a loss looks near-certain).
Signal vs realized return over 200 windows: Pearson r = −0.033 (p = 0.64),
Spearman −0.040. By signal quintile:

| Signal bucket | mean realized return | up-rate |
|---|---:|---:|
| Q1 (most bearish) | **+2.04%** | 70% |
| Q2 | +0.56% | 60% |
| Q3 | +1.68% | 70% |
| Q4 | +1.03% | 60% |
| Q5 (most bullish) | +1.29% | 65% |

The signal is not merely weak — it is slightly **inverted** at the tails. Even
the two most extreme sell signals out of 200 preceded +6.26% average returns,
and the model was *more* bullish than usual before the worst 10% of outcomes
(p = 0.34, i.e. no warning at all). No threshold, asymmetry or sensitivity
setting can extract profit from this.

## 5. Where the model *does* have skill: volatility

Returns are close to unpredictable; volatility is not. Horse-race over 90
non-overlapping (symbol, window) points, SPY/QQQ/TLT (`research/vol_race.py`),
predicting realized volatility of the next 20 trading days:

| method | corr | Spearman | RMSE | QLIKE |
|---|---:|---:|---:|---:|
| **Kronos (path vol)** | **0.382** | 0.535 | **7.77%** | **0.378** |
| Kronos (sample dispersion) | 0.387 | 0.453 | 8.85% | 1.042 |
| Persistence | 0.332 | 0.432 | 9.38% | 0.453 |
| EWMA / RiskMetrics | 0.311 | 0.461 | 8.86% | 0.380 |
| GARCH(1,1) | 0.285 | **0.551** | 8.71% | 0.379 |

Kronos wins on correlation, RMSE and QLIKE — a real but **modest** skill,
essentially GARCH-class. Combining it with GARCH does not help much
(`research/vol_ensemble.py`): the two methods' **errors correlate at 0.85**, so
they carry nearly the same information. Honest caveat: the fine-tuned model saw
these tickers' history during training, so a *historical* win carries a leakage
caveat; a clean confirmation needs forward validation.

## 6. What actually produced an edge (no forecasting model involved)

**6.1 Cross-sectional momentum** over 27 ETFs, monthly, top-5, after costs
(`research/xsec_momentum*.py`, 2004–2026, 272 months):

| | CAGR | Vol | Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| Momentum (raw) | 10.4% | 14.9% | 0.75 | −26.6% |
| Momentum + vol target | 8.0% | 11.0% | 0.75 | −16.4% |
| SPY buy & hold | 10.9% | 14.6% | 0.79 | −50.8% |
| 60/40 (SPY/TLT) | 8.3% | 9.9% | **0.86** | −28.5% |

Momentum alone does *not* beat 60/40 risk-adjusted. But the two win in opposite
regimes (momentum in the GFC, 60/40 in calm bulls), so blending them does
(`research/combo_test.py`): every blend from 25% to 75% momentum beats 60/40,
peaking at **Sharpe 0.91** — robust, not a single tuned point. A *dynamic*
regime tilt on top adds nothing beyond the static blend
(`research/dynamic_tilt.py`).

**6.2 Trend + volatility scaling** — the strongest configuration found
(`research/slope_signal.py`, `research/slope_plus_vol.py`, 10 ETFs, 2006–2026,
248 months). Direction from the trailing 12-month median slope; position size
from a forward volatility estimate:

| | CAGR | Vol | Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| Buy & hold (equal weight) | **9.6%** | 13.4% | 0.76 | **−46.0%** |
| Slope only (long/flat) | 8.6% | 9.5% | 0.92 | −17.7% |
| Vol scaling only | 8.2% | 9.4% | 0.89 | −20.5% |
| **Slope × vol scaling (EWMA)** | 7.3% | 7.5% | **0.99** | −13.4% |
| Slope × vol scaling (GARCH) | 7.1% | 7.3% | 0.98 | **−11.9%** |

The two components are **additive** (0.92 and 0.89 alone → 0.99 together), and
max drawdown falls from −46% to −12%. The cost is real: ~2.3 percentage points
of CAGR per year. Slope lookback matters and behaves exactly as the literature
predicts — 12–24 months work, 36–60 months fail (long-horizon reversal).

Note that the volatility slot is where the Kronos forecast could plug in, but
the gap between good volatility estimators is small (EWMA 0.99 vs GARCH 0.98),
so the expected gain from swapping in a GPU model is marginal.

**6.3 Single stocks are worse, not better** (`research/single_stocks.py`).
Higher crash risk does not mean more value from crash buffering:

| | Sharpe gain from architecture | drawdown cut | CAGR cost | worst 1-day (median) |
|---|---:|---:|---:|---:|
| 30 single stocks | **+0.11** | 40.9 pts | −8.6 pts | −25.6% |
| 10 ETFs | **+0.23** | 32.5 pts | −2.3 pts | −11.7% |

Much of a single stock's extra risk arrives as **overnight gaps**, which a
trend/vol system cannot dodge, while the cost of buffering rises far more.
(Survivorship caveat: these names still trade today, so absolute returns are
inflated for strategy and benchmark alike; the *relative* comparison is the
meaningful one.)

---

## Summary

| Question | Answer |
|---|---|
| Can a fine-tuned foundation model predict ETF direction? | **No** — indistinguishable from a random walk, slightly inverted at the tails |
| Does calibration help? | **Yes** — lifts point forecasts to naive level and yields well-calibrated 80/90% intervals |
| Do lower/zero fees create an edge? | **No** — buy & hold still wins at zero cost |
| Can thresholds or asymmetry rescue it? | **No** — there is no information to threshold |
| Does the model have *any* real skill? | **Yes, volatility** — GARCH-class, modest, leakage caveat |
| What did produce an edge? | **Portfolio construction** — trend + volatility scaling (Sharpe 0.76 → 0.99, drawdown −46% → −12%), and a momentum/60-40 blend (0.86 → 0.91) |

The practical conclusion is the one the efficient-market literature predicts:
on liquid ETFs, effort spent on **risk management and portfolio construction**
pays; effort spent on **return prediction** does not.

## Limitations

- Backtests are historical and idealised; no slippage, market impact, taxes, or
  cash yield on idle capital (cash is modelled at 0%, which understates the
  defensive strategies).
- 200 backtest windows span 10 ETFs at overlapping times → roughly 20
  independent periods, so single-symbol significance is weak.
- Portfolio backtests use today's ETF list (mild survivorship bias) and report
  the best of several configurations, so the top numbers are an optimistic edge
  of the range.
- The fine-tuned model saw 2010–2025 history in training, so historical
  evaluations of *its* skill carry a leakage caveat. Only forward testing is
  fully clean.
