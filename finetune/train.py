from __future__ import annotations

import math
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "finetune" / "config.yaml"
KRONOS_REPO_CANDIDATES = [
    PROJECT_ROOT.parents[1] / "Kronos",
    Path.home() / "Kronos",
]


def status(message: str) -> None:
    print(message, flush=True)


def resolve_kronos_repo() -> Path:
    for candidate in KRONOS_REPO_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "GitHub Kronos repo not found. Expected one of: "
        + ", ".join(str(path) for path in KRONOS_REPO_CANDIDATES)
    )


KRONOS_REPO = resolve_kronos_repo()
if str(KRONOS_REPO) not in sys.path:
    sys.path.insert(0, str(KRONOS_REPO))

from model import Kronos, KronosTokenizer  # noqa: E402


def parse_config_path() -> Path:
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Kronos on a prepared dataset.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to a config.yaml (see config_etf.yaml).")
    args, _ = parser.parse_known_args()
    return Path(args.config)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing PyYAML. Install pyyaml, then rerun train.py.") from exc
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_name = str(config["model"]["name"]).lower()
    tokenizer_name = str(config["model"]["tokenizer_name"]).lower()
    if "amazon" in model_name or "chronos-t5" in model_name:
        raise RuntimeError(f"Amazon Chronos model is not allowed: {config['model']['name']}")
    if "amazon" in tokenizer_name or "chronos-t5" in tokenizer_name:
        raise RuntimeError(f"Amazon Chronos tokenizer is not allowed: {config['model']['tokenizer_name']}")
    return config


def load_arrow(path: str):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("Missing package 'datasets'. Install: pip install datasets") from exc
    return load_from_disk(str(PROJECT_ROOT / path))


def make_time_features(start_iso: str, length: int, bar_step: timedelta = timedelta(hours=1)) -> np.ndarray:
    import pandas as pd

    start = pd.Timestamp(start_iso)
    timestamps = pd.Series([start + bar_step * i for i in range(length)])
    return np.column_stack(
        [
            timestamps.dt.minute.to_numpy(),
            timestamps.dt.hour.to_numpy(),
            timestamps.dt.weekday.to_numpy(),
            timestamps.dt.day.to_numpy(),
            timestamps.dt.month.to_numpy(),
        ]
    ).astype(np.float32)


class KronosWindowDataset(Dataset):
    def __init__(
        self,
        arrow_data,
        context_length: int,
        prediction_length: int,
        clip: float = 5.0,
        bar_step: timedelta = timedelta(hours=1),
    ) -> None:
        self.data = arrow_data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_length = context_length + prediction_length
        self.clip = clip
        self.bar_step = bar_step

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.data[int(idx)]
        if "features" not in record:
            raise RuntimeError("Fine-tune data is missing Kronos 'features'. Rerun finetune/prepare_data.py.")

        features = np.asarray(record["features"], dtype=np.float32)[: self.window_length]
        if features.ndim != 2 or features.shape[1] != 6:
            raise RuntimeError(f"Expected Kronos feature shape (T, 6), got {features.shape}")
        if features.shape[0] < self.window_length:
            raise RuntimeError(f"Expected at least {self.window_length} rows, got {features.shape[0]}")

        context = features[: self.context_length]
        mean = context.mean(axis=0)
        std = context.std(axis=0)
        normalized = (features - mean) / (std + 1e-5)
        normalized = np.clip(normalized, -self.clip, self.clip).astype(np.float32)
        stamps = make_time_features(str(record["start"]), len(normalized), self.bar_step)
        return {
            "x": torch.from_numpy(normalized),
            "stamp": torch.from_numpy(stamps),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "stamp": torch.stack([item["stamp"] for item in batch]),
    }


def configure_vram(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        return
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb < 6.5:
        status("[WARNING] <=6GB VRAM detected. Using batch_size=4 and gradient_accumulation_steps=8.")
        config["training"]["batch_size"] = min(int(config["training"]["batch_size"]), 4)
        config["training"]["gradient_accumulation_steps"] = max(int(config["training"]["gradient_accumulation_steps"]), 8)


def build_models(config: dict[str, Any], device: torch.device) -> tuple[KronosTokenizer, Kronos]:
    tokenizer_name = config["model"]["tokenizer_name"]
    model_name = config["model"]["name"]
    status(f"[TRAIN] BASE MODEL: {model_name}")
    output_dir = PROJECT_ROOT / config["training"]["output_dir"]
    tokenizer_cfg = config.get("tokenizer", {})
    tokenizer_candidates = [
        output_dir / tokenizer_cfg.get("save_folder", "tokenizer_finetuned"),
        output_dir / tokenizer_cfg.get("quicktest_folder", "tokenizer_quicktest"),
    ]
    tokenizer_dir = next((path for path in tokenizer_candidates if path.exists()), None)
    if tokenizer_dir is not None:
        status(f"[TRAIN] Loading fine-tuned tokenizer: {tokenizer_dir}")
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir)).to(device).eval()
    else:
        status("[TRAIN] No fine-tuned tokenizer found - using base tokenizer.")
        status(f"[TRAIN] TOKENIZER : {tokenizer_name}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name).to(device).eval()
    model = Kronos.from_pretrained(model_name).to(device)
    assert "amazon" not in str(model_name).lower(), "Wrong model loaded: Amazon Chronos is not allowed"
    return tokenizer, model


def kronos_loss(tokenizer: KronosTokenizer, model: Kronos, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    x = batch["x"].to(device, non_blocking=True)
    stamp = batch["stamp"].to(device, non_blocking=True)
    with torch.no_grad():
        token_s1, token_s2 = tokenizer.encode(x, half=True)

    token_in = [token_s1[:, :-1], token_s2[:, :-1]]
    token_out = [token_s1[:, 1:], token_s2[:, 1:]]
    logits = model(token_in[0], token_in[1], stamp[:, :-1, :])
    loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
    return loss


def evaluate_on_split(tokenizer: KronosTokenizer, model: Kronos, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            losses.append(float(kronos_loss(tokenizer, model, batch, device).detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else math.inf


def save_checkpoint(model: Kronos, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)


def main() -> None:
    config = load_config(parse_config_path())
    configure_vram(config)
    bar_step = timedelta(hours=1) if config["model"].get("interval", "1h") == "1h" else timedelta(days=1)

    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_data = load_arrow(config["data"]["train_path"])
    val_data = load_arrow(config["data"]["val_path"])
    context_length = int(config["model"]["context_length"])
    prediction_length = int(config["model"]["prediction_length"])

    train_ds = KronosWindowDataset(train_data, context_length, prediction_length, bar_step=bar_step)
    val_ds = KronosWindowDataset(val_data, context_length, prediction_length, bar_step=bar_step)
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    tokenizer, model = build_models(config, device)
    model.train()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    total_steps = max(1, math.ceil(len(train_loader) / accumulation) * int(config["training"]["num_epochs"]))
    warmup_steps = int(config["training"]["warmup_steps"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    output_dir = PROJECT_ROOT / config["training"]["output_dir"]
    best_dir = output_dir / "best_clean"
    best_val = math.inf
    global_step = 0

    for epoch in range(int(config["training"]["num_epochs"])):
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            loss = kronos_loss(tokenizer, model, batch, device) / accumulation
            loss.backward()
            epoch_losses.append(float(loss.detach().cpu()) * accumulation)

            if batch_idx % accumulation == 0 or batch_idx == len(train_loader):
                clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_idx % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[TRAIN] epoch={epoch + 1}/{config['training']['num_epochs']} "
                    f"batch={batch_idx}/{len(train_loader)} loss={epoch_losses[-1]:.4f} lr={lr:.2e}",
                    flush=True,
                )

        val_loss = evaluate_on_split(tokenizer, model, val_loader, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else math.inf
        print(f"Epoch {epoch + 1} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}", flush=True)

        if (epoch + 1) % int(config["training"]["save_every_n_epochs"]) == 0:
            save_checkpoint(model, output_dir / f"epoch_{epoch + 1}")

        if val_loss < best_val:
            best_val = val_loss
            if best_dir.exists():
                shutil.rmtree(best_dir)
            save_checkpoint(model, best_dir)

    status(f"Training complete. Best val_loss={best_val:.4f}. Saved: {best_dir}")


if __name__ == "__main__":
    main()
