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
from typing import Any, Dict, Literal, Optional


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
class OptionPositionJson:
    """A held contract. Every field the equity shapes have no room for.

    `collateral` is the number that matters and the one a share-shaped view
    cannot show: a short put obliges us to buy 100 shares at the strike, so a
    $92 put is a $9,200 commitment while its market value reads as $120.
    Rendering only market_value understates the position by two orders of
    magnitude.
    """

    symbol: str
    underlying: str
    right: Literal["call", "put"]
    strike: float
    expiry: str            # ISO yyyy-mm-dd
    dte: int
    qty: float             # negative when short
    side: Literal["long", "short"]
    market_value: float
    unrealized_pl: float
    collateral: float      # cash posted; 0 for long options and short calls
    moneyness_pct: float | None   # distance to the strike, percent; None if unquoted
    itm: bool | None


@dataclass
class OptionPlanJson:
    """One decision from the overlay, including the ones that traded nothing.

    The refusals are the point. A dashboard that only lists fills cannot show
    the agent declining for a stated reason, which is most of what it does.
    """

    ts: str
    summary: str
    equity: float
    collateral_posted: float
    open_options: int
    plans: list[dict]
    orders: list[dict]
    skipped: list[dict]


@dataclass
class OptionsConfigJson:
    """The limits, so the UI can render a cap rather than an unexplained refusal."""

    enabled: bool
    target_delta: float
    delta_min: float
    delta_max: float
    max_collateral_pct: float   # percent (50.0 = 50%)
    max_positions: int
    max_orders_per_day: int
    min_credit: float


@dataclass
class AdvisorVerdictJson:
    """One LLM review, and how the underlying then moved.

    `correct` is deliberately nullable and means three things, not two: True,
    False, or "the market did not move enough to say". Collapsing the third
    into False would score noise as a mistake.
    """

    ts: str
    symbol: str
    side: str                       # buy | sell
    action: str                     # confirm | shrink | veto
    reason: str
    price_at: float
    move_pct: Optional[float]       # None until settled
    correct: Optional[bool]


@dataclass
class AdvisorStatsJson:
    """Accuracy with its sample size. One without the other is unreadable --
    100% of two calls and 100% of forty are the same number and different facts."""

    overall_rate: float
    overall_n: int
    veto_rate: float
    veto_n: int
    pending: int                    # recorded but not yet judged
    model: str


@dataclass
class AllocationJson:
    """How the account is split between the two sides right now."""

    equity_budget: float
    options_budget: float
    reserve: float
    regime: str
    reason: str
    enabled: bool


@dataclass
class HealthJson:
    status: Literal["ok"]
    mode: Literal["PAPER", "LIVE"]


SignalsJson = Dict[str, str]  # symbol -> "BUY" | "SELL" | "HOLD"