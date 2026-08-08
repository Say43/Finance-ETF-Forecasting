# Finance-ETF-Forecasting

Fine-tuning, conformal calibration, and rigorous evaluation of the
[Kronos](https://github.com/shiyu-coder/Kronos) financial foundation model on
ETF daily data — plus everything that followed once the honest evaluation said
"no directional edge."

## Goal

Take a pretrained candlestick foundation model, adapt it to ETFs, calibrate its
output properly, and then evaluate it as rigorously as a research paper would
— including negative results. The project ended up answering a broader
question along the way: **is there any way to turn this model into a tradable
edge**, and if not, **what actually does work** for a systematically managed
ETF portfolio.

**tl;dr of the findings** (full detail in
[docs/results.md](docs/results.md)):

- The fine-tuned model's **directional forecasts are indistinguishable from a
  random walk** on liquid ETFs — and slightly *inverted* at the extremes. No
  fee level, threshold, or asymmetric rule turns this into an edge.
- Its **volatility forecasts do have real (if modest) skill**, roughly on par
  with GARCH(1,1).
- The things that *did* produce a better risk-adjusted portfolio needed no
  forecasting model at all: trailing-trend direction combined with
  volatility-scaled position sizing took a 10-ETF equal-weight portfolio from
  Sharpe 0.76 (max drawdown −46%) to Sharpe 0.99 (max drawdown −12%).

This repository is the honest record of that investigation, not a trading
product. Read [DISCLAIMER.md](DISCLAIMER.md).

## Methodology

1. **Data.** Free daily OHLCV for US ETFs via Yahoo Finance (`yfinance`), NYSE
   trading calendar via `pandas-market-calendars`. No paid data anywhere.
2. **Fine-tuning.** `NeoQuasar/Kronos-small` fine-tuned on a 22-ETF universe
   (3 held out as an out-of-sample test set), 360-day context, 20-day horizon,
   with a low-LR/early-stopping recipe that fixed a v1 overfitting problem.
3. **Calibration.** A model-agnostic post-processing layer — bias correction,
   forecast combination (shrinkage toward a naive/drift blend), and split
   conformal prediction intervals — fitted on leakage-free walk-forward
   windows.
4. **Evaluation.** Walk-forward backtests with a Diebold–Mariano significance
   test against the naive baseline, volatility forecasting scored against
   GARCH/EWMA with the proper QLIKE loss, and — because the headline
   directional result was negative — a battery of follow-up tests (zero fees,
   asymmetric thresholds, one month of realistic paper trading) to make sure
   the negative result was real and not an artifact.
5. **Portfolio research.** Independent, model-free tests of cross-sectional
   momentum, trend following, and volatility targeting, to find out what
   *does* produce a real, defensible edge on the same asset class.

Full methodology: [docs/methodology.md](docs/methodology.md). Full results
with tables: [docs/results.md](docs/results.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Fine-tuning and inference need a local clone of the upstream Kronos model code
(MIT-licensed, not vendored here):

```bash
git clone https://github.com/shiyu-coder/Kronos.git
export KRONOS_REPO=/path/to/Kronos    # Windows: set KRONOS_REPO=...
```

Full reproduction steps (fine-tune → calibrate → evaluate → web UI):
[docs/reproduce.md](docs/reproduce.md).

## Data

Everything is fetched at runtime from free, public endpoints — nothing is
shipped in this repository. See [NOTICE.md](NOTICE.md) for data-provider terms
and third-party attributions.

## Project structure

```
src/kronos_etf/         Core package: asset config, calibration, model I/O
finetune/                Fine-tuning pipeline (prepare_data.py, train.py,
                          config, Kaggle notebook)
scripts/                 CLI entry points: backtest.py, calibrate.py, demo.py
research/                Standalone portfolio-construction experiments
                          (momentum, trend + vol targeting, single-stock test)
webapp/                  Local Flask UI for interactive forecasts
tests/                   pytest suite (calibration math)
docs/                    methodology.md, results.md, reproduce.md
```

## Results

See [docs/results.md](docs/results.md) for full tables, including the negative
directional-forecasting result, the volatility-forecasting horse race, and the
portfolio-construction experiments that did produce a real improvement.

## License

MIT — see [LICENSE](LICENSE). Third-party code, model weights, and data-source
terms are documented in [NOTICE.md](NOTICE.md). This is research software, not
investment advice — see [DISCLAIMER.md](DISCLAIMER.md).
