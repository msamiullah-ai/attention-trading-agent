"""Configuration loading: credentials from .env, tunables from config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Credentials:
    api_key: str
    secret_key: str
    paper: bool

    @property
    def mode(self) -> str:
        return "PAPER" if self.paper else "LIVE"


@dataclass
class RiskConfig:
    # Position sizing: Kelly-derived fraction of equity, scaled by this
    # multiplier (quarter=0.25, half=0.5, full=1.0) and always capped below.
    kelly_multiplier: float = 0.5
    # Hard ceiling on any single trade's size, regardless of what Kelly says.
    max_position_pct: float = 0.05
    max_open_positions: int = 3
    max_daily_loss_pct: float = 0.03
    # Trades required before trusting a live win-rate/R-ratio for Kelly sizing.
    min_trades_for_kelly: int = 20
    # Flat position size used before min_trades_for_kelly is reached.
    fallback_position_pct: float = 0.02
    # Window of recent trades checked for the negative-expectancy pause.
    expectancy_window: int = 20


@dataclass
class ExecutionConfig:
    order_type: str = "market"
    time_in_force: str = "day"
    limit_slippage_pct: float = 0.001
    require_market_open: bool = True


@dataclass
class EngineConfig:
    poll_interval_seconds: int = 60


@dataclass
class DataConfig:
    feed: str = "iex"


@dataclass
class Config:
    symbols: list[str]
    strategy: str
    strategy_params: dict[str, Any] = field(default_factory=dict)
    timeframe: str = "1Min"
    lookback_bars: int = 250
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


def load_credentials(env_path: Path | None = None) -> Credentials:
    """Read Alpaca credentials from .env / environment.

    Raises RuntimeError with actionable text rather than letting the SDK fail
    with an opaque 403 later on.
    """
    load_dotenv(env_path or PROJECT_ROOT / ".env")

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() not in {"false", "0", "no"}

    missing = [
        name
        for name, value in (("ALPACA_API_KEY", api_key), ("ALPACA_SECRET_KEY", secret_key))
        if not value or value.startswith("your_")
    ]
    if missing:
        raise RuntimeError(
            f"Missing credentials: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your Alpaca keys."
        )

    return Credentials(api_key=api_key, secret_key=secret_key, paper=paper)


def load_config(path: Path | None = None) -> Config:
    """Load config.yaml into typed dataclasses."""
    path = path or PROJECT_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    symbols = [s.strip().upper() for s in raw.get("symbols", []) if s.strip()]
    if not symbols:
        raise ValueError(f"No symbols configured in {path}")

    return Config(
        symbols=symbols,
        strategy=raw.get("strategy", "ema_rsi"),
        strategy_params=raw.get("strategy_params") or {},
        timeframe=raw.get("timeframe", "1Min"),
        lookback_bars=int(raw.get("lookback_bars", 250)),
        data=DataConfig(**(raw.get("data") or {})),
        risk=RiskConfig(**(raw.get("risk") or {})),
        execution=ExecutionConfig(**(raw.get("execution") or {})),
        engine=EngineConfig(**(raw.get("engine") or {})),
    )
