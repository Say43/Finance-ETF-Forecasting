# Reproducing this project

## 1. Prerequisites

- Python 3.11+
- A CUDA GPU is recommended for fine-tuning and inference (a 6GB card is
  enough for `Kronos-small`); everything also runs on CPU, just slower.
- A local clone of the upstream [Kronos](https://github.com/shiyu-coder/Kronos)
  model code (MIT-licensed, not vendored here — see [NOTICE.md](../NOTICE.md)):

  ```bash
  git clone https://github.com/shiyu-coder/Kronos.git
  export KRONOS_REPO=/path/to/Kronos      # Windows: set KRONOS_REPO=...
  ```

  If unset, the code also looks for `../Kronos` and `~/Kronos`.

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                        # installs the kronos_etf package
```

## 3. Fine-tune (optional — a checkpoint isn't shipped, see NOTICE.md)

```bash
python finetune/prepare_data.py --asset-class etf
python finetune/train.py --config finetune/config_etf.yaml
```

This is the local-GPU path. The Kaggle notebook
(`finetune/kronos_etf_finetune.ipynb`) reproduces the same recipe end-to-end on
a free T4 GPU and was the one actually used to produce the results in
[docs/results.md](results.md); push it with the Kaggle CLI
(`kaggle kernels push -p finetune`) or run it interactively on kaggle.com.

Recipe (v2): 22-ETF universe, LR 3e-5 with cosine decay, step-level early
stopping (patience 4), EMA weights, fp16 mixed precision, wall-clock guard.
See `docs/methodology.md` and the notebook for exact hyperparameters.

## 4. Calibrate

```bash
python scripts/calibrate.py --asset-class etf \
    --symbols SPY,QQQ,IWM,DIA,EFA,GLD,TLT --n-windows 15 --use-finetuned
```

Writes `finetune/checkpoints_etf/calibration_etf.json`.

## 5. Evaluate

```bash
python scripts/backtest.py --symbol SPY --asset-class etf --use-finetuned
python scripts/demo.py --symbol SPY --asset-class etf --use-finetuned
```

## 6. Portfolio research scripts

Each script in `research/` is standalone and downloads its own data on demand;
no GPU or fine-tuned checkpoint is required for the portfolio-construction
scripts (`slope_signal.py`, `slope_plus_vol.py`, `xsec_momentum*.py`,
`combo_test.py`, `dynamic_tilt.py`, `single_stocks.py`). The Kronos-dependent
ones (`vol_race.py`, `simulate_month.py`, `bear_test.py`, `resim.py`) need a
fine-tuned checkpoint and calibration file from steps 3–4.

```bash
python research/slope_plus_vol.py
```

## 7. Web UI

```bash
python webapp/app.py
```

Then open `http://127.0.0.1:5000`. Needs a fine-tuned checkpoint and
calibration file.

## 8. Tests

```bash
pytest tests/
```

## Notes on determinism

Fine-tuning uses a fixed seed (`42`) but exact reproduction across different
GPU/driver/PyTorch versions is not guaranteed (standard cuDNN non-determinism).
Backtests and portfolio scripts are deterministic given the same market-data
snapshot; Yahoo Finance data can be revised after the fact (e.g. corporate
action adjustments), so re-running on a later date may shift results slightly.
