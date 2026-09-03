"""Unit checks for the read-only API contract. Network-free and key-free.

Pins the units the frontend depends on (the off-by-100 fraction/percent bugs
from the first UI PR) plus the signal HOLD rule and the equity curve
shape. No FastAPI test client, no Alpaca credentials: these only exercise
the pure converters and route registration from api.py.
"""

from types import SimpleNamespace

import pytest

from api import (
    Runtime,
    account_to_json,
    app,
    pick_signal,
    position_to_json,
    stats_to_json,
    trades_to_curve,
)

# ------------------------------------------------------------------- account


class FakeAccount:
    equity = 104250.32
    cash = 18320.11
    buying_power = 36640.22
    last_equity = 102790.0
    blocked = False

    @property
    def daily_pnl_pct(self) -> float:
        return (self.equity - self.last_equity) / self.last_equity


def test_account_fields_are_percent_not_fraction():
    out = account_to_json(FakeAccount(), "PAPER")
    assert out.mode == "PAPER"
    assert out.blocked is False
    assert out.equity == 104250.32
    # 1460.32 / 102790.0 = 0.0142068...  which must come out as 1.42%, not 0.0142%
    assert out.daily_pnl_pct == pytest.approx(1.4207, abs=1e-3)


# ----------------------------------------------------------------- positions


class FakePosition:
    symbol = "AAPL"
    qty = 40.0
    market_value = 9412.8
    avg_entry_price = 221.15
    unrealized_pl = 348.6
    unrealized_plpc = 0.0385
    side = "long"


def test_position_plpc_is_percent_and_side_kept():
    out = position_to_json(FakePosition())  # type: ignore[arg-type]
    assert out.symbol == "AAPL"
    assert out.unrealized_plpc == pytest.approx(3.85)
    assert out.side == "long"


# ------------------------------------------------------------------- signals


class FakeStrategy:
    min_bars = 21

    def generate_signal(self, df):  # noqa: D102
        return "BUY"


def test_pick_signal_returns_hold_without_enough_bars():
    assert pick_signal(FakeStrategy(), None) == "HOLD"
    assert pick_signal(FakeStrategy(), range(5)) == "HOLD"


def test_pick_signal_delegates_when_enough_bars():
    assert pick_signal(FakeStrategy(), range(21)) == "BUY"


# -------------------------------------------------------------- risk metrics


class FakeStats:
    n = 12
    win_rate = 0.6
    loss_rate = 0.4
    avg_win = 0.0
    avg_loss = 0.0
    avg_r_winner = 0.0
    reward_risk_ratio = 0.0
    expectancy = 0.0
    expectancy_r = 0.25
    breakeven_win_rate = 0.5
    margin_of_safety = 0.1

    @property
    def has_data(self) -> bool:
        return self.n > 0


def test_risk_rates_are_percent_and_expectancy_is_in_r():
    out = stats_to_json(FakeStats(), kelly_position_pct=0.05, strategy="ema_rsi", window=50)
    assert out.n == 12
    assert out.win_rate == pytest.approx(60.0)
    assert out.loss_rate == pytest.approx(40.0)
    assert out.breakeven_win_rate == pytest.approx(50.0)
    assert out.margin_of_safety == pytest.approx(10.0)
    assert out.expectancy == pytest.approx(0.25)  # R per trade, not a percent
    assert out.kelly_position_pct == pytest.approx(5.0)
    assert out.strategy == "ema_rsi"
    assert out.window == 50


# -------------------------------------------------------------- equity curve


def test_equity_curve_is_cumulative_and_date_formatted():
    risk = SimpleNamespace(
        trade_log=SimpleNamespace(
            trades=[
                SimpleNamespace(pnl=100.0, exit_time="2026-08-30T15:00:00+00:00"),
                SimpleNamespace(pnl=-25.0, exit_time="2026-08-31T15:00:00+00:00"),
            ]
        )
    )
    curve = trades_to_curve(risk)  # type: ignore[arg-type]
    assert [point.equity for point in curve] == [100.0, 75.0]
    assert [point.date for point in curve] == ["08/30", "08/31"]


# ----------------------------------------------------------------- freshness


def test_fresh_risk_rereads_trade_log_per_call():
    """The bot appends closed trades to state/trades.csv from its own process.

    TradeLog only reads that file in its constructor, so caching one
    RiskManager on the Runtime would freeze the equity curve and expectancy
    at whatever had closed when the first request came in.
    """
    runtime = Runtime(
        creds=SimpleNamespace(mode="PAPER"),  # type: ignore[arg-type]
        cfg=SimpleNamespace(risk=SimpleNamespace()),  # type: ignore[arg-type]
        broker=SimpleNamespace(),  # type: ignore[arg-type]
        data=SimpleNamespace(),  # type: ignore[arg-type]
        strategy=SimpleNamespace(),  # type: ignore[arg-type]
    )
    first, second = runtime.fresh_risk(), runtime.fresh_risk()
    assert first is not second
    assert first.trade_log is not second.trade_log


# ------------------------------------------------------------------ routes


def test_all_routes_registered():
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    for expected in (
        "/health",
        "/api/account",
        "/api/positions",
        "/api/signals",
        "/api/equity-curve",
        "/api/risk-metrics",
    ):
        assert expected in paths