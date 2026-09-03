"""Tests for signal fusion inside the trade path.

test_agent.py covers the fusion arithmetic in isolation. These cover the parts
that only exist once it is plugged in: who is allowed to vote, what an
abstention does, and the interaction with the boolean regime gate -- which is
the one place where turning fusion on has to REMOVE an existing check rather
than add one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_trader import (  # noqa: E402
    FakeBroker, FakeMarketData, FakeStrategy, make_config, make_creds, make_df,
)
from trading_bot.risk import RiskManager, TradeLog  # noqa: E402
from trading_bot.signals import BUY, HOLD, REGISTRY, SELL  # noqa: E402
from trading_bot.trader import Trader, _build_ensemble  # noqa: E402


def long_df(n=120, start=100.0, step=0.5):
    """Enough bars that every strategy clears min_bars, with a real trend so
    the indicators produce something other than a flat line."""
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    close = [start + i * step for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": [c + 0.5 for c in close],
         "low": [c - 0.5 for c in close], "close": close, "volume": [1000] * n},
        index=idx,
    )


def uncorrelated_df(seed: int, n=120):
    """An independent random walk per symbol, so the correlation guard is not
    silently doing the work a cap test means to measure."""
    import numpy as np
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": [1000] * n}, index=idx)


def build(fusion=None, strategy=None, bars=None, broker=None, **kw):
    config = make_config()
    broker = broker or FakeBroker()
    return broker, Trader(
        config, make_creds(), broker=broker,
        data=FakeMarketData(bars=bars or {"AAPL": make_df()}),
        risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
        strategy=strategy or FakeStrategy(signal=BUY), fusion=fusion, **kw,
    )


# ---------------------------------------------------------------- default off

def test_fusion_off_by_default(monkeypatch):
    monkeypatch.delenv("TRADING_BOT_FUSION", raising=False)
    _, trader = build()
    assert trader.ensemble == {}


def test_fusion_off_uses_the_configured_strategy_alone(monkeypatch):
    monkeypatch.delenv("TRADING_BOT_FUSION", raising=False)
    broker, trader = build()
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1     # unchanged behaviour


def test_env_var_turns_it_on(monkeypatch):
    monkeypatch.setenv("TRADING_BOT_FUSION", "true")
    _, trader = build()
    assert len(trader.ensemble) == len(REGISTRY)


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_env_var_only_accepts_affirmatives(monkeypatch, value):
    monkeypatch.setenv("TRADING_BOT_FUSION", value)
    _, trader = build()
    assert trader.ensemble == {}


# ----------------------------------------------------------------- ensemble

def test_ensemble_builds_every_registered_strategy():
    assert set(_build_ensemble("ema_rsi", None)) == set(REGISTRY)


def test_only_the_primary_receives_configured_params():
    """Passing one strategy's tuning to another is a TypeError at best."""
    ens = _build_ensemble("ema_rsi", {"ema_period": 5})
    assert ens["ema_rsi"].min_bars != ens["bollinger_squeeze"].min_bars
    # bollinger_squeeze must have been built on defaults, not on ema_period
    assert "bollinger_squeeze" in ens


def test_unknown_primary_still_builds_the_rest():
    """FakeStrategy's name is not in REGISTRY; that must not break anything."""
    assert set(_build_ensemble("fake", None)) == set(REGISTRY)


# ----------------------------------------------------------- who may vote

def test_short_history_makes_every_strategy_abstain():
    """5 bars is under every real strategy's min_bars."""
    _, trader = build(fusion=True)
    assert trader._collect_votes("AAPL", make_df(5)) == {}


def test_abstention_is_absence_not_hold():
    """The distinction that keeps a strategy which never ran from voting."""
    _, trader = build(fusion=True)
    votes = trader._collect_votes("AAPL", make_df(5))
    assert HOLD not in votes.values()
    assert votes == {}


def test_long_history_lets_strategies_vote():
    _, trader = build(fusion=True)
    votes = trader._collect_votes("AAPL", long_df())
    assert len(votes) >= 2
    assert all(v in (BUY, SELL, HOLD) for v in votes.values())


def test_crypto_momentum_excluded_for_equities():
    _, trader = build(fusion=True)
    assert "crypto_momentum" not in trader._collect_votes("AAPL", long_df())


def test_crypto_momentum_included_for_crypto():
    _, trader = build(fusion=True)
    assert "crypto_momentum" in trader._collect_votes("BTC/USD", long_df(200))


def test_a_raising_strategy_abstains_rather_than_crashing():
    class Exploding(FakeStrategy):
        name = "boom"
        min_bars = 1

        def generate_signal(self, df):
            raise RuntimeError("indicator blew up")

    _, trader = build(fusion=True)
    trader.ensemble["boom"] = Exploding()
    votes = trader._collect_votes("AAPL", long_df())
    assert "boom" not in votes
    assert len(votes) >= 2          # the others still voted


# ------------------------------------------------------- regime gate change

def test_regime_gate_still_applies_without_fusion(monkeypatch):
    """A strategy out of its regime is silenced when it decides alone."""
    import trading_bot.trader as tr
    monkeypatch.setattr(tr, "detect_regime", lambda df: "BULL")
    strat = FakeStrategy(signal=BUY)
    strat.name = "vwap_mean_reversion"          # SIDEWAYS only
    broker, trader = build(fusion=False, strategy=strat)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_regime_gate_is_skipped_under_fusion(monkeypatch):
    """Removing it is the point: the weights already applied the same table
    continuously, so re-applying it as a boolean would undo fusion."""
    import trading_bot.trader as tr
    monkeypatch.setattr(tr, "detect_regime", lambda df: "BULL")
    strat = FakeStrategy(signal=BUY)
    strat.name = "vwap_mean_reversion"
    broker, trader = build(fusion=True, strategy=strat)
    monkeypatch.setattr(trader, "_collect_votes",
                        lambda s, d: {"ema_rsi": BUY, "bollinger_squeeze": BUY})
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1


# --------------------------------------------------------------- end to end

def test_unanimous_buy_reaches_the_broker(monkeypatch):
    broker, trader = build(fusion=True)
    monkeypatch.setattr(trader, "_collect_votes",
                        lambda s, d: {"ema_rsi": BUY, "bollinger_squeeze": BUY})
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1
    assert broker.submitted_orders[0]["side"] == "buy"


def test_split_vote_sends_nothing(monkeypatch):
    broker, trader = build(fusion=True, fusion_temperature=1000.0)
    monkeypatch.setattr(trader, "_collect_votes", lambda s, d: {
        "ema_rsi": BUY, "bollinger_squeeze": SELL})
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_verdict_is_recorded_for_inspection(monkeypatch):
    _, trader = build(fusion=True)
    monkeypatch.setattr(trader, "_collect_votes", lambda s, d: {"ema_rsi": BUY})
    trader.run_cycle()
    assert "AAPL" in trader.last_verdict
    assert trader.last_verdict["AAPL"].weights


def test_empty_votes_yield_hold_and_no_order(monkeypatch):
    broker, trader = build(fusion=True)
    monkeypatch.setattr(trader, "_collect_votes", lambda s, d: {})
    trader.run_cycle()
    assert trader.last_signal["AAPL"] == HOLD
    assert broker.submitted_orders == []


def test_primary_strategy_still_sets_the_levels(monkeypatch):
    """Ensemble decides direction; the primary decides stop and target."""
    strat = FakeStrategy(signal=HOLD, stop=88.0, target=112.0)
    broker, trader = build(fusion=True, strategy=strat)
    monkeypatch.setattr(trader, "_collect_votes", lambda s, d: {"ema_rsi": BUY})
    trader.run_cycle()
    order = broker.submitted_orders[0]
    assert order["stop_loss_price"] == 88.0
    assert order["take_profit_price"] == 112.0


# ------------------------------------------------- position cap within a cycle

def test_max_open_positions_holds_within_one_cycle(monkeypatch):
    """`positions` is fetched once per cycle. Without reserving each new entry,
    every later symbol sees the pre-cycle count and the cap does nothing --
    one cycle could submit far more orders than max_open_positions allows."""
    config = make_config()
    config.symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    config.risk.max_open_positions = 2
    # Correlation must not be what stops us here, or the test would pass while
    # the cap did nothing. Identical series correlate at 1.0 and get blocked by
    # the correlation guard, so each symbol gets its own uncorrelated walk.
    config.risk.correlation_threshold = 1.01
    broker = FakeBroker()
    bars = {s: uncorrelated_df(seed=i) for i, s in enumerate(config.symbols)}
    trader = Trader(
        config, make_creds(), broker=broker, data=FakeMarketData(bars=bars),
        risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
        strategy=FakeStrategy(signal=BUY), fusion=False,
    )
    trader.run_cycle()
    assert len(broker.submitted_orders) == 2, (
        f"cap is 2 but {len(broker.submitted_orders)} orders were sent")


def test_dry_run_reports_the_capped_count_too(monkeypatch, caplog):
    """A dry run that ignores the cap advertises trades the bot would refuse."""
    import logging
    config = make_config()
    config.symbols = ["AAA", "BBB", "CCC", "DDD"]
    config.risk.max_open_positions = 1
    config.risk.correlation_threshold = 1.01
    broker = FakeBroker()
    trader = Trader(
        config, make_creds(), broker=broker,
        data=FakeMarketData(bars={s: uncorrelated_df(seed=i)
                                  for i, s in enumerate(config.symbols)}),
        risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
        strategy=FakeStrategy(signal=BUY), dry_run=True, fusion=False,
    )
    with caplog.at_level(logging.INFO):
        trader.run_cycle()
    would = [r for r in caplog.records if "DRY RUN" in r.getMessage()]
    assert len(would) == 1
