# Third-party notices, data terms and attribution

This project is MIT-licensed (see [LICENSE](LICENSE)). It builds on third-party
work and public data sources listed below. Nothing in this repository
redistributes third-party model weights or market data.

## Upstream model: Kronos

The forecasting model is [Kronos](https://github.com/shiyu-coder/Kronos), a
foundation model for financial candlestick series, released by its authors under
the **MIT License**. This repository:

- **does not vendor** the Kronos source code — it is loaded from a local clone
  you provide via the `KRONOS_REPO` environment variable, so the upstream code
  stays under its own license and version control;
- **does not redistribute** the pretrained weights
  (`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-Tokenizer-base`), which are
  downloaded at runtime from the Hugging Face Hub under their own terms;
- **does not commit** the fine-tuned checkpoint produced by this pipeline. It is
  a derivative of the MIT-licensed base weights trained on publicly available
  price data, and is reproducible with `finetune/` (see
  [docs/reproduce.md](docs/reproduce.md)).

If you use Kronos in academic work, the authors ask that you cite their paper:

> Kronos: A Foundation Model for the Language of Financial Markets (2025).
> https://github.com/shiyu-coder/Kronos

## Market data

Price data is retrieved at runtime from **Yahoo Finance** via the
[`yfinance`](https://github.com/ranaroussi/yfinance) library (Apache-2.0).

Important terms of use:

- `yfinance` is an **unofficial** client. It is not affiliated with, endorsed
  by, or certified by Yahoo. Yahoo Finance data is intended for **personal,
  non-commercial use** and is subject to Yahoo's own terms of service.
- This repository ships **no market data**. All datasets, caches and downloaded
  files are gitignored; every script fetches data on demand for local research.
- Users are responsible for complying with the data provider's terms and for
  obtaining an appropriately licensed data feed for any commercial use.

Trading-calendar data comes from
[`pandas-market-calendars`](https://github.com/rsheftel/pandas_market_calendars)
(MIT).

## Python dependencies

| Package | License |
|---|---|
| PyTorch | BSD-3-Clause |
| NumPy, pandas, SciPy | BSD-3-Clause |
| Hugging Face `transformers`, `datasets`, `safetensors`, `huggingface_hub` | Apache-2.0 |
| `yfinance` | Apache-2.0 |
| `arch` (GARCH baselines) | NCSA |
| `pandas-market-calendars`, `einops`, `PyYAML`, `matplotlib`, `Flask`, `pytest` | MIT / BSD / PSF-compatible |

All are permissively licensed and compatible with redistributing this project
under the MIT License. No copyleft (GPL/AGPL) dependencies are used.

## Benchmarks and prior work referenced

Baselines and evaluation choices follow standard practice in the forecasting and
empirical-finance literature (GARCH(1,1), EWMA/RiskMetrics volatility,
random-walk benchmarks, Diebold–Mariano tests, split conformal prediction,
time-series and cross-sectional momentum). See
[docs/methodology.md](docs/methodology.md) for the specific formulations used.
