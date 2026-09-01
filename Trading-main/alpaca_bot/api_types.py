"""JSON contract shared with the frontend. Mirror these shapes in the TS types.

Unit conventions (the single source of truth that fixes the old off-by-100
display bug):

  - *_pct fields are PERCENT, not fractions: `daily_pnl_pct=1.42` means 1.42%.
  - `side` is "long"/"short" (never Alpaca's l/s letters).
  - `expectancy` is R-multiples per trade, not a percent.
  - equity curve points are cumulative realized P/L from closed trades,
    oldest first; an empty list means no trades have closed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class AccountJson:
    equity: float
    cash: float
    buying_power: float
    daily_pnl_pct: float  # percent (1.42 = 1.42%)
    mode: Literal["PAPER", "LIVE"]
    blocked: bool


@dataclass
class PositionJson:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float  # percent (-3.4 = -3.4%)
    side: Literal["long", "short"]


@dataclass
class EquityPointJson:
    date: str  # MM/DD
    equity: float


@dataclass
class RiskMetricsJson:
    n: int
    win_rate: float            # percent
    loss_rate: float           # percent
    expectancy: float          # R-multiples per trade
    breakeven_win_rate: float  # percent
    margin_of_safety: float    # percent
    kelly_position_pct: float  # percent, the capped size the bot actually uses
    strategy: str
    window: int


@dataclass
class HealthJson:
    status: Literal["ok"]
    mode: Literal["PAPER", "LIVE"]


SignalsJson = Dict[str, str]  # symbol -> "BUY" | "SELL" | "HOLD"