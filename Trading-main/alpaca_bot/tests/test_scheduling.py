"""Tests for Deficit Round Robin slot allocation.

Two properties have to hold at once and they pull against each other: the best
signal should win now, and a symbol that keeps losing should win eventually.
A scheduler with only the first is what file order already gave us; one with
only the second ignores merit entirely. Most of these pin the balance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.scheduling import (  # noqa: E402
    DEFAULT_MAX_DEFICIT, DEFAULT_QUANTUM, Candidate, DeficitRoundRobin,
)
from trading_bot.signals import BUY  # noqa: E402


def c(symbol, score):
    return Candidate(symbol=symbol, score=score, signal=BUY)


# ------------------------------------------------------------------ merit

def test_best_signal_wins_immediately():
    picked = DeficitRoundRobin().select([c("A", 0.5), c("B", 0.9), c("C", 0.6)], 1)
    assert [x.symbol for x in picked] == ["B"]


def test_fills_every_available_slot():
    picked = DeficitRoundRobin().select([c("A", 0.5), c("B", 0.9), c("C", 0.6)], 2)
    assert [x.symbol for x in picked] == ["B", "C"]


def test_never_exceeds_the_slots_offered():
    picked = DeficitRoundRobin().select([c(s, 0.5) for s in "ABCDE"], 2)
    assert len(picked) == 2


def test_no_slots_selects_nothing():
    assert DeficitRoundRobin().select([c("A", 0.9)], 0) == []


def test_no_candidates_is_not_an_error():
    assert DeficitRoundRobin().select([], 3) == []


# ------------------------------------------------- the bug this replaces

def test_list_order_does_not_decide():
    """The whole point. Under the old loop the first qualifying symbol in
    config.yaml won regardless of signal strength -- measured live, slots went
    to indices 649/871/892 and everything after was refused."""
    early_weak = c("AAPL", 0.55)      # index 0 in a real config
    late_strong = c("ZUMZ", 0.85)     # index 997
    picked = DeficitRoundRobin().select([early_weak, late_strong], 1)
    assert picked[0].symbol == "ZUMZ"


def test_ties_break_on_name_not_position():
    """Equal scores must not silently fall back to input order, or file
    position leaks back in through the sort."""
    forward = DeficitRoundRobin().select([c("B", 0.5), c("A", 0.5)], 1)
    reverse = DeficitRoundRobin().select([c("A", 0.5), c("B", 0.5)], 1)
    assert forward[0].symbol == reverse[0].symbol == "A"


# ------------------------------------------------------------ starvation

def test_a_passed_over_symbol_accrues_credit():
    s = DeficitRoundRobin()
    s.select([c("A", 0.9), c("B", 0.5)], 1)
    assert s.deficit["B"] == pytest.approx(DEFAULT_QUANTUM)
    assert s.deficit["A"] == 0.0        # winner had none to spend


def test_persistent_loser_eventually_wins():
    """Merit first, but not merit forever."""
    s = DeficitRoundRobin()
    winners = []
    for _ in range(20):
        picked = s.select([c("A", 0.70), c("B", 0.55)], 1)
        winners.append(picked[0].symbol)
    assert "B" in winners, "B was starved for 20 consecutive cycles"


def test_a_big_enough_gap_is_never_overturned():
    """Fairness must not become its own unfairness: the deficit is capped, so a
    genuinely far better signal keeps winning."""
    s = DeficitRoundRobin()
    for _ in range(200):
        picked = s.select([c("A", 0.99), c("B", 0.10)], 1)
    assert picked[0].symbol == "A"


def test_deficit_is_capped():
    s = DeficitRoundRobin()
    for _ in range(500):
        s.select([c("A", 0.9), c("B", 0.1)], 1)
    assert s.deficit["B"] == DEFAULT_MAX_DEFICIT


def test_winning_spends_credit_rather_than_zeroing_it():
    """A symbol that waited a long time then traded should start the next
    contest even, not back at the bottom."""
    s = DeficitRoundRobin()
    s.deficit["B"] = 0.30
    s.select([c("B", 0.9)], 1)
    assert s.deficit["B"] == pytest.approx(0.30 - DEFAULT_QUANTUM)


def test_nobody_accrues_when_there_were_no_slots():
    """Credit is for being passed over, not for a cycle in which the book was
    simply full."""
    s = DeficitRoundRobin()
    s.select([c("A", 0.5), c("B", 0.5)], 0)
    assert s.deficit == {}


# --------------------------------------------------------------- wiring

def test_config_rejects_an_unknown_scheduler(tmp_path):
    import yaml
    from trading_bot.config import load_config
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "symbols": ["AAPL"], "strategy": "ema_rsi", "timeframe": "1Min",
        "lookback_bars": 250, "execution": {"scheduler": "nonsense"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid scheduler"):
        load_config(p)


def test_scheduler_defaults_to_off():
    from trading_bot.config import ExecutionConfig
    assert ExecutionConfig().scheduler == "none"


def test_trader_builds_no_scheduler_by_default():
    from test_trader import (FakeBroker, FakeMarketData, FakeStrategy,
                             make_config, make_creds, make_df)
    from trading_bot.risk import RiskManager, TradeLog
    from trading_bot.trader import Trader
    config = make_config()
    t = Trader(config, make_creds(), broker=FakeBroker(),
               data=FakeMarketData(bars={"AAPL": make_df()}),
               risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
               strategy=FakeStrategy(signal=BUY))
    assert t.scheduler is None


def test_trader_respects_the_slot_budget_with_drr():
    """End to end: more qualifying symbols than slots, and only the budget is
    filled -- with the choice made on merit rather than on list position."""
    import numpy as np
    import pandas as pd
    from test_trader import FakeBroker, FakeMarketData, FakeStrategy, make_config, make_creds
    from trading_bot.risk import RiskManager, TradeLog
    from trading_bot.trader import Trader

    def walk(seed):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 1, 60))
        idx = pd.date_range("2024-01-02 09:30", periods=60, freq="1min", tz="UTC")
        return pd.DataFrame({"open": close, "high": close + .5, "low": close - .5,
                             "close": close, "volume": [1000] * 60}, index=idx)

    config = make_config()
    config.symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    config.risk.max_open_positions = 2
    config.risk.correlation_threshold = 1.01      # isolate the cap from correlation
    config.execution.scheduler = "drr"
    broker = FakeBroker()
    trader = Trader(config, make_creds(), broker=broker,
                    data=FakeMarketData(bars={s: walk(i) for i, s in enumerate(config.symbols)}),
                    risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
                    strategy=FakeStrategy(signal=BUY))
    assert trader.scheduler is not None
    trader.run_cycle()
    assert len(broker.submitted_orders) == 2


# ---------------------------------------------------------------- metrics

def test_metrics_are_empty_before_anything_runs():
    m = DeficitRoundRobin().metrics()
    assert m["slots_awarded"] == 0 and m["fairness"] == 1.0


def test_max_wait_is_the_starvation_indicator():
    """The number that matters: it climbs without bound only if some symbol is
    never getting through, which is the failure this scheduler prevents."""
    s = DeficitRoundRobin()
    for _ in range(30):
        s.select([c("A", 0.70), c("B", 0.55)], 1)
    assert s.metrics()["max_wait"] < 30, "B was never served"


def test_a_genuinely_starved_symbol_shows_up_in_max_wait():
    """The metric has to be able to report failure, or it is decoration."""
    s = DeficitRoundRobin()
    for _ in range(40):
        s.select([c("A", 0.99), c("B", 0.05)], 1)   # gap exceeds the deficit cap
    assert s.metrics()["max_wait"] >= 30


def test_fairness_is_one_when_slots_are_shared_evenly():
    s = DeficitRoundRobin()
    for _ in range(10):
        s.select([c("A", 0.5), c("B", 0.5)], 2)     # both served every cycle
    assert s.metrics()["fairness"] == pytest.approx(1.0)


def test_fairness_drops_when_one_symbol_takes_everything():
    s = DeficitRoundRobin()
    for _ in range(40):
        s.select([c("A", 0.99), c("B", 0.05)], 1)
    m = s.metrics()
    assert m["fairness"] < 0.7
    assert m["symbols_served"] == 1.0


def test_avg_wait_counts_cycles_not_awards():
    s = DeficitRoundRobin()
    s.select([c("A", 0.9), c("B", 0.5)], 1)   # B waits 1
    s.select([c("A", 0.9), c("B", 0.5)], 1)   # B waits 2
    s.select([c("B", 0.9)], 1)                # B served after waiting 2
    assert s.wait_on_service[-1] == 2


def test_serving_resets_that_symbols_wait():
    s = DeficitRoundRobin()
    s.select([c("A", 0.9), c("B", 0.5)], 1)
    assert s.waits["B"] == 1
    s.select([c("B", 0.9)], 1)
    assert s.waits["B"] == 0
