"""Tests for Opening Range Breakout.

The rules are cheap to state and easy to get subtly wrong, so each one gets a
test that fails if it is removed: the range must freeze, unconfirmed breaks
must be ignored, a level must not be re-entered after it fails, and the stop
must sit on the correct side of entry. A strategy that trades twice a day has
no room for a rule that only mostly works.
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.signals import OpeningRangeBreakoutStrategy  # noqa: E402
from trading_bot.signals import BUY, HOLD, SELL  # noqa: E402


def session(bars, start="2026-09-03 09:30", volume=1000):
    """Build one US session of 1-minute bars from (high, low, close) triples."""
    idx = pd.date_range(start, periods=len(bars), freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"open": [b[2] for b in bars], "high": [b[0] for b in bars],
         "low": [b[1] for b in bars], "close": [b[2] for b in bars],
         "volume": [volume] * len(bars)},
        index=idx.tz_convert("UTC"),
    )


def flat_then(after, n_range=20, base=100.0, volume=1000, range_volume=1000):
    """A quiet 20-minute opening range around `base`, then `after` bars."""
    opening = [(base + 0.5, base - 0.5, base)] * n_range
    df = session(opening + after)
    vols = [range_volume] * n_range + [volume] * len(after)
    df["volume"] = vols
    return df


def strat(**kw):
    return OpeningRangeBreakoutStrategy(range_minutes=15, volume_period=10, **kw)


def signals(s, df):
    ctx = s.precompute(df)
    return [s.evaluate(df, ctx, i) for i in range(len(df))]


# ------------------------------------------------------------- the range

def test_no_signal_while_the_range_is_forming():
    s = strat()
    df = flat_then([])
    assert set(signals(s, df)) == {HOLD}


def test_breakout_above_the_range_goes_long():
    s = strat()
    df = flat_then([(102.0, 101.0, 101.5)], volume=5000)
    assert BUY in signals(s, df)


def test_breakdown_below_the_range_goes_short():
    s = strat()
    df = flat_then([(99.0, 98.0, 98.5)], volume=5000)
    assert SELL in signals(s, df)


def test_price_inside_the_range_does_nothing():
    s = strat()
    df = flat_then([(100.2, 99.8, 100.0)], volume=5000)
    assert set(signals(s, df)) == {HOLD}


def test_a_session_with_too_few_opening_bars_is_skipped():
    """Half days and gappy data must not produce a range that never existed."""
    s = strat()
    df = session([(101.0, 99.0, 100.0)])
    assert set(signals(s, df)) == {HOLD}


# ------------------------------------------------------ the volume filter

def test_an_unconfirmed_break_is_ignored():
    """The dominant ORB failure mode: price pokes through on no volume and
    reverts. Without this filter the strategy bleeds."""
    s = strat()
    df = flat_then([(102.0, 101.0, 101.5)], volume=100, range_volume=1000)
    assert BUY not in signals(s, df)


def test_the_same_break_on_volume_does_fire():
    """The mirror of the test above -- proves the filter is what blocked it,
    not something else about the bar."""
    s = strat()
    quiet = flat_then([(102.0, 101.0, 101.5)], volume=100, range_volume=1000)
    loud = flat_then([(102.0, 101.0, 101.5)], volume=9000, range_volume=1000)
    assert BUY not in signals(s, quiet)
    assert BUY in signals(s, loud)


# ------------------------------------------------------------ one per side

def test_a_failed_level_is_not_re_entered():
    """One choppy session must not become ten losses."""
    s = strat()
    chop = [(102.0, 101.0, 101.5), (100.2, 99.8, 100.0)] * 5
    assert signals(s, flat_then(chop, volume=5000)).count(BUY) == 1


def test_both_sides_can_trade_in_one_session():
    s = strat()
    df = flat_then([(102.0, 101.0, 101.5), (99.0, 98.0, 98.5)], volume=5000)
    out = signals(s, df)
    assert BUY in out and SELL in out


# --------------------------------------------------------------- the cutoff

def test_no_entries_after_the_cutoff():
    """A breakout with twenty minutes left cannot reach a sensible target."""
    s = strat(no_entry_after=time(9, 50))
    df = flat_then([(102.0, 101.0, 101.5)] * 10, volume=5000)
    assert BUY not in signals(s, df)


# ---------------------------------------------------------------- the stop

def test_stop_is_the_far_side_of_the_range():
    """Not an ATR multiple: the range IS the thesis, so price returning through
    it means the reason for the trade is gone."""
    s = strat()
    df = flat_then([(102.0, 101.0, 101.5)], volume=5000)
    assert s.get_stop_loss(101.5, df, "long") == pytest.approx(99.5)


def test_short_stop_is_the_range_high():
    s = strat()
    df = flat_then([(99.0, 98.0, 98.5)], volume=5000)
    assert s.get_stop_loss(98.5, df, "short") == pytest.approx(100.5)


def test_a_stop_on_the_wrong_side_of_entry_is_rejected():
    """Alpaca rejects it, and a 'stop' above a long entry is not a stop."""
    s = strat()
    df = flat_then([(102.0, 101.0, 101.5)], volume=5000)
    stop = s.get_stop_loss(99.0, df, "long")     # entry below the range low
    assert stop < 99.0


def test_no_range_falls_back_rather_than_disabling_the_stop():
    s = strat()
    df = session([(101.0, 99.0, 100.0)])
    assert s.get_stop_loss(100.0, df, "long") == pytest.approx(99.0)


def test_target_is_a_multiple_of_the_range_risk():
    s = strat(target_r=2.0)
    df = flat_then([(102.0, 101.0, 101.5)], volume=5000)
    entry = 101.5
    stop = s.get_stop_loss(entry, df, "long")           # 99.5 -> R = 2.0
    assert s.get_take_profit(entry, df, "long") == pytest.approx(entry + 2 * (entry - stop))


# --------------------------------------------------------------- plumbing

def test_context_is_not_shared_between_runs():
    """`taken` on self instead of in the context would let a backtest silently
    suppress signals in a later live run."""
    s = strat()
    df = flat_then([(102.0, 101.0, 101.5)], volume=5000)
    assert signals(s, df).count(BUY) == 1
    assert signals(s, df).count(BUY) == 1        # second run unaffected


def test_it_is_in_the_registry_and_config_accepts_it():
    from trading_bot.config import VALID_STRATEGIES
    from trading_bot.signals import REGISTRY, build_strategy
    assert "opening_range_breakout" in REGISTRY
    assert "opening_range_breakout" in VALID_STRATEGIES
    assert build_strategy("opening_range_breakout").allow_short is True


def test_generate_signal_uses_the_last_bar():
    s = strat()
    df = flat_then([(102.0, 101.0, 101.5)], volume=5000)
    assert s.generate_signal(df) == BUY


# ------------------------------------------------------------- conviction

def test_base_strategy_is_neutral_by_default():
    """A strategy with no notion of conviction should say so rather than
    invent a number that looks like information."""
    from trading_bot.signals import EmaRsiStrategy
    assert EmaRsiStrategy().strength(flat_then([])) == 0.5


def test_a_decisive_break_scores_above_a_marginal_one():
    """The whole reason this exists: without it every candidate ties and the
    scheduler falls back to alphabetical order."""
    s = strat()
    marginal = s.strength(flat_then([(100.6, 100.5, 100.55)], volume=5000))
    decisive = s.strength(flat_then([(101.0, 100.6, 100.9)], volume=5000))
    assert decisive > marginal


def test_higher_volume_scores_higher_at_the_same_extension():
    s = strat()
    thin = s.strength(flat_then([(101.0, 100.6, 100.9)], volume=1100, range_volume=1000))
    heavy = s.strength(flat_then([(101.0, 100.6, 100.9)], volume=4000, range_volume=1000))
    assert heavy > thin


def test_decisive_break_on_no_volume_is_still_weak():
    """Multiplied not averaged: either factor missing means a weak signal."""
    s = strat()
    assert s.strength(flat_then([(101.0, 100.6, 100.9)], volume=1000, range_volume=1000)) < 0.2


def test_strength_never_returns_zero():
    """It passed every gate to get here; zero would let the deficit alone rank
    it, which is the tie problem again."""
    s = strat()
    assert s.strength(flat_then([(100.51, 100.49, 100.5)], volume=1000)) >= 0.05


def test_strength_is_bounded():
    s = strat()
    v = s.strength(flat_then([(200.0, 150.0, 180.0)], volume=999999))
    assert 0.0 <= v <= 1.0


def test_no_range_is_neutral_not_confident():
    s = strat()
    assert s.strength(session([(101.0, 99.0, 100.0)])) == 0.5
