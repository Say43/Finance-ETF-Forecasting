from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssetConfig:
    """Everything that differs between asset classes (crypto vs. ETF)."""

    symbol: str
    asset_class: str  # "crypto" | "etf"
    interval: str  # e.g. "1h" for crypto, "1d" for ETFs
    feature_columns: tuple[str, ...]
    calendar: str | None = None  # None => 24/7 (crypto); e.g. "XNYS" for US equities/ETFs
    step: object = None  # step size for one bar; set in __post_init__
    max_step_pct: tuple[float, float] = (0.012, 0.018)  # legacy fixed clip, unused when vol-adaptive
    regime_thresholds: tuple[float, float] = (-0.10, -0.02)  # (crash, bear) 7-bar-window pct move
    lookback: int = 360
    pred_len: int = 48
    checkpoint_path: str = "finetune/checkpoints/best_clean"

    def __post_init__(self) -> None:
        from datetime import timedelta

        step = timedelta(hours=1) if self.interval == "1h" else timedelta(days=1)
        object.__setattr__(self, "step", step)

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto"

    @property
    def is_etf(self) -> bool:
        return self.asset_class == "etf"


CRYPTO_FEATURE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "funding_rate")
ETF_FEATURE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# US-listed ETFs used for multi-symbol fine-tuning (broad, liquid, and covering
# different exposures so the model doesn't just memorize one price path).
ETF_FINETUNE_UNIVERSE: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA", "EFA", "GLD", "TLT")


def get_asset_config(symbol: str, asset_class: str) -> AssetConfig:
    asset_class = asset_class.lower()
    if asset_class == "crypto":
        return AssetConfig(
            symbol=symbol,
            asset_class="crypto",
            interval="1h",
            feature_columns=CRYPTO_FEATURE_COLUMNS,
            calendar=None,
            regime_thresholds=(-0.10, -0.02),
            lookback=360,
            pred_len=48,
            checkpoint_path="finetune/checkpoints/best_clean",
        )
    if asset_class == "etf":
        return AssetConfig(
            symbol=symbol,
            asset_class="etf",
            interval="1d",
            feature_columns=ETF_FEATURE_COLUMNS,
            calendar="XNYS",
            # ETFs move far less than BTC; crash/bear thresholds over a
            # 7-bar (7 trading day) window are tightened accordingly.
            regime_thresholds=(-0.07, -0.015),
            lookback=360,
            pred_len=20,
            checkpoint_path="finetune/checkpoints_etf/best_clean",
        )
    raise ValueError(f"Unknown asset_class: {asset_class!r} (expected 'crypto' or 'etf')")
