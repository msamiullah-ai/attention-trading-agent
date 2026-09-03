"""Tests for the learning loop: closed trades changing future weights.

The risk here is subtle in both directions. A loop that learns too eagerly
thrashes -- two bad trades and a strategy is effectively switched off for the
day. One that never learns is the state this replaced. Most of these tests pin
the middle: the record has to move the weights, but slowly, and never to zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.attention import attend  # noqa: E402
from trading_bot.learning import FLOOR, PRIOR, StrategyMemory  # noqa: E402
from trading_bot.signals import BUY, HOLD, SELL  # noqa: E402


def mem():
    return StrategyMemory(in_memory=True)


def feed(m, strategy, wins, losses, direction=BUY):
    for i in range(wins):
        m.record({strategy: direction}, direction, "AAA", r_multiple=1.5)
    for i in range(losses):
        m.record({strategy: direction}, direction, "AAA", r_multiple=-1.0)


# --------------------------------------------------------------- cold start

def test_cold_start_is_neutral():
    """With no history nothing may be penalised, or the bot cannot begin."""
    assert mem().reliability() == {}


def test_unknown_strategy_defaults_to_full_weight():
    """attend() treats a missing entry as 1.0; a new strategy is not suspect."""
    m = mem()
    feed(m, "ema_rsi", wins=0, losses=5)
    rel = m.reliability()
    assert "bollinger_squeeze" not in rel
    v = attend({"bollinger_squeeze": BUY}, regime="BULL", reliability=rel)
    assert v.action == BUY


# ------------------------------------------------------------- credit rules

def test_only_voters_in_the_taken_direction_are_scored():
    """A strategy that said HOLD did not cause the trade."""
    m = mem()
    n = m.record({"a": BUY, "b": HOLD, "c": SELL}, BUY, "AAA", r_multiple=1.0)
    assert n == 1
    assert set(m.hit_rates()) == {"a"}


def test_a_trade_with_no_votes_teaches_nothing():
    """Positions opened before fusion was on carry no votes; not an error."""
    m = mem()
    assert m.record({}, BUY, "AAA", r_multiple=1.0) == 0
    assert m.hit_rates() == {}


def test_win_and_loss_are_split_on_r_multiple():
    m = mem()
    feed(m, "a", wins=3, losses=1)
    rate, n = m.hit_rates()["a"]
    assert n == 4 and rate == pytest.approx(0.75)


def test_breakeven_counts_as_a_loss():
    """r_multiple of exactly 0 paid commission and spread for nothing."""
    m = mem()
    m.record({"a": BUY}, BUY, "AAA", r_multiple=0.0)
    assert m.hit_rates()["a"][0] == 0.0


# ------------------------------------------------------------- the shrinkage

def test_two_losses_barely_move_the_weight():
    """The anti-thrash property: a bad morning is not evidence."""
    m = mem()
    feed(m, "a", wins=0, losses=2)
    assert m.reliability()["a"] > 0.8


def test_a_sustained_bad_record_does_move_it():
    m = mem()
    feed(m, "a", wins=2, losses=38)
    assert m.reliability()["a"] < 0.5


def test_reliability_never_reaches_zero():
    """A silenced strategy cannot warn you the one time it is right."""
    m = mem()
    feed(m, "a", wins=0, losses=200)
    assert m.reliability()["a"] == FLOOR


def test_a_winning_streak_buys_no_extra_volume():
    """Capped at 1.0: a hot streak is the easiest thing to mistake for skill."""
    m = mem()
    feed(m, "a", wins=40, losses=0)
    assert m.reliability()["a"] == 1.0


def test_coin_flip_record_is_neutral():
    m = mem()
    feed(m, "a", wins=20, losses=20)
    assert m.reliability()["a"] == pytest.approx(1.0)


# ------------------------------------------------- it actually changes votes

def test_a_bad_record_can_flip_the_fused_decision():
    """The whole point: the record has to reach the decision, not just a file."""
    votes = {"ema_rsi": BUY, "vwap_mean_reversion": SELL}
    before = attend(votes, regime="BULL", temperature=0.5)
    assert before.action == BUY          # ema_rsi suits BULL and wins on fit

    m = mem()
    feed(m, "ema_rsi", wins=1, losses=39)
    after = attend(votes, regime="BULL", temperature=0.5, reliability=m.reliability())
    assert after.weights["ema_rsi"] < before.weights["ema_rsi"]


def test_the_quieted_strategy_still_has_a_voice():
    m = mem()
    feed(m, "ema_rsi", wins=0, losses=100)
    v = attend({"ema_rsi": BUY, "vwap_mean_reversion": SELL}, regime="BULL",
               reliability=m.reliability())
    assert v.weights["ema_rsi"] > 0


# ------------------------------------------------------------- the window

def test_only_recent_outcomes_count():
    """Old results describe a market that no longer exists."""
    m = StrategyMemory(in_memory=True, window=10)
    feed(m, "a", wins=0, losses=50)      # ancient history
    feed(m, "a", wins=10, losses=0)      # recent form
    assert m.hit_rates()["a"][0] == 1.0


# ------------------------------------------------------------- persistence

def test_round_trips_through_disk(tmp_path):
    p = tmp_path / "scores.jsonl"
    m = StrategyMemory(path=p)
    feed(m, "a", wins=3, losses=1)
    assert StrategyMemory(path=p).hit_rates()["a"] == (pytest.approx(0.75), 4)


def test_a_torn_final_line_is_ignored(tmp_path):
    """A killed process can leave a half-written row; it must not poison the
    whole record."""
    p = tmp_path / "scores.jsonl"
    m = StrategyMemory(path=p)
    feed(m, "a", wins=2, losses=0)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"strategy": "a", "wo')
    assert StrategyMemory(path=p).hit_rates()["a"][1] == 2


def test_an_unwritable_path_does_not_stop_trading(tmp_path):
    """Losing a score degrades the weighting; it must not break the decision."""
    m = StrategyMemory(path=tmp_path / "nope" / "x" / "scores.jsonl")
    m.path = tmp_path            # a directory: open() for append will fail
    m.record({"a": BUY}, BUY, "AAA", r_multiple=1.0)
    assert m.hit_rates()["a"][1] == 1          # in-memory still correct


def test_summary_reports_sample_size_not_just_rate():
    """100% of two and 100% of forty are the same number, different facts."""
    m = mem()
    feed(m, "a", wins=2, losses=0)
    assert "(2)" in m.summary()


# ------------------------------------------------- end to end through Trader

def test_trader_settles_votes_when_a_position_closes(monkeypatch):
    """The loop through the real Trader: open with votes, close, learn.

    Asserted through Trader rather than on StrategyMemory alone because the
    fragile part is the handoff -- votes have to survive from the verdict onto
    the OpenTrade and back out at finalisation. A memory that works in
    isolation while the trader never calls it is the bug this guards."""
    from test_trader import (FakeBroker, FakeMarketData, FakeStrategy,
                             make_config, make_creds, make_df)
    from trading_bot.risk import RiskManager, TradeLog
    from trading_bot.trader import Trader

    config = make_config()
    broker = FakeBroker()
    m = StrategyMemory(in_memory=True)
    trader = Trader(config, make_creds(), broker=broker,
                    data=FakeMarketData(bars={"AAPL": make_df()}),
                    risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
                    strategy=FakeStrategy(signal=BUY), fusion=True, memory=m)
    # The BUY voter must be the one that SUITS the regime, or the fused score
    # lands inside the deadband and no trade opens at all. 5 bars read as
    # SIDEWAYS, which is vwap_mean_reversion's regime and not ema_rsi's.
    monkeypatch.setattr(trader, "_collect_votes",
                        lambda s, d: {"vwap_mean_reversion": BUY, "ema_rsi": HOLD})

    trader.run_cycle()
    assert "AAPL" in trader.open_trades
    assert trader.open_trades["AAPL"].votes == {"vwap_mean_reversion": BUY,
                                                "ema_rsi": HOLD}

    # The position disappears from the broker -> the trade is finalised.
    broker.positions.clear()
    trader.run_cycle()

    rates = m.hit_rates()
    assert "vwap_mean_reversion" in rates, "the BUY voter was never credited"
    assert "ema_rsi" not in rates, "a HOLD voter must not be scored"
