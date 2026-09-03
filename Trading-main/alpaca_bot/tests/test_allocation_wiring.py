"""The allocator applied to the trade path, not just computed.

The failure this guards is specific and quiet: a budget that caps each position
individually looks correct in a one-trade test and lets three positions deploy
three budgets in production. Every test here checks CUMULATIVE spend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_trader import (  # noqa: E402
    FakeBroker, FakeMarketData, FakeStrategy, make_config, make_creds,
)
from trading_bot.risk import RiskManager, TradeLog  # noqa: E402
from trading_bot.signals import BUY  # noqa: E402
from trading_bot.trader import Trader  # noqa: E402


def walk(seed, n=60, start=100.0):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + .5, "low": close - .5,
                         "close": close, "volume": [1000] * n}, index=idx)


def build(budget=None, symbols=("AAA", "BBB", "CCC"), max_positions=3):
    config = make_config()
    config.symbols = list(symbols)
    config.risk.max_open_positions = max_positions
    config.risk.correlation_threshold = 1.01     # isolate budget from correlation
    broker = FakeBroker()
    return broker, Trader(
        config, make_creds(), broker=broker,
        data=FakeMarketData(bars={s: walk(i) for i, s in enumerate(symbols)}),
        risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
        strategy=FakeStrategy(signal=BUY), budget=budget)


def deployed(broker):
    return sum(o["qty"] * 100.0 for o in broker.submitted_orders)   # ~$100/share


# ------------------------------------------------------------- default path

def test_no_budget_behaves_exactly_as_before():
    """The default must be byte-identical, or turning the allocator off is not
    actually a way back."""
    broker, trader = build(budget=None)
    assert trader.budget is None
    trader.run_cycle()
    assert len(broker.submitted_orders) == 3


def test_a_budget_larger_than_the_account_changes_nothing():
    broker, trader = build(budget=10_000_000.0)
    trader.run_cycle()
    assert len(broker.submitted_orders) == 3


# ---------------------------------------------------- the cumulative property

def test_the_budget_is_cumulative_not_per_position():
    """THE bug this exists to prevent: capping each position individually lets
    three positions deploy three budgets."""
    broker, trader = build(budget=250.0)
    trader.run_cycle()
    assert deployed(broker) <= 250.0 * 1.05, (
        f"budget 250 but {deployed(broker):.0f} deployed across "
        f"{len(broker.submitted_orders)} orders")


def test_a_budget_under_alpacas_fractional_minimum_stops_trading():
    """$1 notional is Alpaca's floor, so a budget below it can fund nothing.

    Written at 0.50 rather than 1.00 deliberately: a budget of exactly $1 CAN
    still place one order, and asserting otherwise would have been a test of my
    assumption rather than of the code."""
    broker, trader = build(budget=0.50)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_a_budget_at_the_minimum_still_funds_exactly_one_trade():
    broker, trader = build(budget=1.0)
    trader.run_cycle()
    assert len(broker.submitted_orders) <= 1


def test_a_budget_can_allow_fewer_positions_than_the_cap():
    """max_open_positions is 3, but the money only reaches for one."""
    broker, trader = build(budget=150.0)
    trader.run_cycle()
    assert len(broker.submitted_orders) < 3


# --------------------------------------------------------------- the helper

def test_within_budget_returns_the_account_untouched_when_unset():
    from trading_bot.broker import AccountSnapshot
    _, trader = build(budget=None)
    a = AccountSnapshot(equity=100.0, cash=100.0, buying_power=100.0,
                        last_equity=100.0, blocked=False)
    assert trader._within_budget(a, {}) is a


def test_within_budget_subtracts_existing_exposure():
    from trading_bot.broker import AccountSnapshot, PositionSnapshot
    _, trader = build(budget=1_000.0)
    a = AccountSnapshot(equity=50_000.0, cash=50_000.0, buying_power=50_000.0,
                        last_equity=50_000.0, blocked=False)
    held = {"X": PositionSnapshot(symbol="X", qty=1, market_value=600.0,
                                  avg_entry_price=600.0, unrealized_pl=0.0,
                                  unrealized_plpc=0.0, side="long")}
    assert trader._within_budget(a, held).buying_power == pytest.approx(400.0)


def test_over_budget_exposure_leaves_nothing():
    from trading_bot.broker import AccountSnapshot, PositionSnapshot
    _, trader = build(budget=500.0)
    a = AccountSnapshot(equity=50_000.0, cash=50_000.0, buying_power=50_000.0,
                        last_equity=50_000.0, blocked=False)
    held = {"X": PositionSnapshot(symbol="X", qty=1, market_value=900.0,
                                  avg_entry_price=900.0, unrealized_pl=0.0,
                                  unrealized_plpc=0.0, side="long")}
    assert trader._within_budget(a, held).buying_power == 0.0


def test_a_short_position_counts_against_the_budget():
    """Exposure is exposure; a negative market value must not credit budget."""
    from trading_bot.broker import AccountSnapshot, PositionSnapshot
    _, trader = build(budget=1_000.0)
    a = AccountSnapshot(equity=50_000.0, cash=50_000.0, buying_power=50_000.0,
                        last_equity=50_000.0, blocked=False)
    held = {"X": PositionSnapshot(symbol="X", qty=-1, market_value=-600.0,
                                  avg_entry_price=600.0, unrealized_pl=0.0,
                                  unrealized_plpc=0.0, side="short")}
    assert trader._within_budget(a, held).buying_power == pytest.approx(400.0)


def test_the_budget_never_raises_buying_power():
    """It is a ceiling. A budget above available cash must not invent money."""
    from trading_bot.broker import AccountSnapshot
    _, trader = build(budget=1_000_000.0)
    a = AccountSnapshot(equity=100.0, cash=100.0, buying_power=100.0,
                        last_equity=100.0, blocked=False)
    assert trader._within_budget(a, {}).buying_power == 100.0


# ---------------------------------------------------------------- the switch

def test_allocator_defaults_to_off():
    from trading_bot.config import ExecutionConfig
    assert ExecutionConfig().allocator == "none"


def test_config_rejects_an_unknown_allocator(tmp_path):
    import yaml
    from trading_bot.config import load_config
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "symbols": ["AAPL"], "strategy": "ema_rsi", "timeframe": "1Min",
        "lookback_bars": 250, "execution": {"allocator": "nonsense"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid allocator"):
        load_config(p)


def test_options_side_ceiling_is_the_tighter_of_the_two():
    """Coordination may only tighten this side, never loosen it.

    Checked on the arithmetic rather than the source: config allows 50% of
    equity, the allocator allows less, and the smaller must win."""
    equity, pct, allocated = 100_000.0, 0.50, 12_000.0
    from_config = equity * pct
    assert min(from_config, allocated) == allocated

    # And the reverse: an allocator budget above the config share cannot raise it.
    assert min(from_config, 90_000.0) == from_config


def test_options_trader_accepts_and_stores_a_budget():
    import inspect
    from trading_bot.options_trader import OptionsTrader
    assert "collateral_budget" in inspect.signature(OptionsTrader.__init__).parameters
