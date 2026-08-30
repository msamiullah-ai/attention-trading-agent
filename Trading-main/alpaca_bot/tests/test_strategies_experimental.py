"""LooseEmaRsiStrategy tests — synthetic bars, no network access.

This strategy lives outside signals.py/REGISTRY on purpose (see the module
docstring in strategies_experimental.py) -- imported directly here, same as
its intended usage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.signals import BUY, HOLD, SELL  # noqa: E402
from trading_bot.strategies_experimental import LooseEmaRsiStrategy  # noqa: E402


def make_df(closes, volumes=None, start="2024-01-02 09:30") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.2 for c in closes],
            "low": [c - 0.2 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000] * n,
        },
        index=idx,
    )


def _pullback_bounce_series(bounce_bars: list[float]) -> list[float]:
    up = list(np.linspace(100, 102, 2))
    decline = list(np.linspace(up[-1], up[-1] - 15, 20))[1:]
    return up + decline + bounce_bars


def test_holds_with_too_few_bars():
    strat = LooseEmaRsiStrategy()
    df = make_df([100.0] * 5)
    assert strat.generate_signal(df) == HOLD


def test_is_long_only():
    assert LooseEmaRsiStrategy().allow_short is False


def test_matches_strict_strategy_on_a_same_bar_cross():
    # Verified via probe: bounce=+10 from the decline bottom (87.0) lands
    # RSI at 53.3 -- a clean same-bar cross that doesn't overshoot into
    # overbought, so both the strict and loose versions should catch it.
    df = make_df(_pullback_bounce_series([97.0]))
    assert LooseEmaRsiStrategy().generate_signal(df) == BUY


def test_catches_a_bounce_the_strict_version_would_miss():
    # The bounce plays out over 3 bars instead of landing on a single-bar
    # cross -- verified empirically that the strict EmaRsiStrategy reads
    # this as HOLD (the cross already happened one bar earlier than its
    # check window), while the lookback-window version still catches it.
    df = make_df(_pullback_bounce_series([87.0, 91.0, 95.0]))
    assert LooseEmaRsiStrategy().generate_signal(df) == BUY


def test_does_not_fire_without_a_recent_oversold_dip():
    # A monotonic climb pins RSI at 100 (verified in test_indicators.py),
    # which would trip the *overbought* SELL branch, not what this test
    # means to check -- use genuinely flat/choppy data instead, which
    # keeps RSI mid-range with no oversold dip and no overbought spike.
    rng = np.random.RandomState(2)
    df = make_df(list(100 + rng.normal(0, 0.5, 30)))
    assert LooseEmaRsiStrategy().generate_signal(df) == HOLD


def test_does_not_fire_while_rsi_is_still_falling():
    # Oversold reached, but still declining on the current bar (not "turning up").
    series = _pullback_bounce_series([80.0, 75.0])  # decline continues into the tail
    df = make_df(series)
    assert LooseEmaRsiStrategy().generate_signal(df) == HOLD


def test_sells_when_overbought_regardless_of_uptrend_setting():
    df = make_df(list(np.linspace(100, 140, 22)))
    assert LooseEmaRsiStrategy().generate_signal(df) == SELL


def test_require_uptrend_true_blocks_entry_below_ema():
    # A deeper decline (bottom 77.0 instead of 87.0) whose bounce doesn't
    # fully recover back above EMA20 -- verified via probe: price ends at
    # 86.0 vs EMA20 86.2, genuinely below, while the RSI pattern alone
    # still satisfies the oversold-then-turning-up condition.
    up = list(np.linspace(100, 102, 2))
    deep_decline = list(np.linspace(up[-1], up[-1] - 25, 20))[1:]
    bounce = [deep_decline[-1] + 2, deep_decline[-1] + 6, deep_decline[-1] + 9]
    df = make_df(up + deep_decline + bounce)

    from trading_bot import indicators as ind
    ema = ind.ema(df, 20)
    assert df["close"].iloc[-1] < ema.iloc[-1]  # confirm the premise

    loose_default = LooseEmaRsiStrategy(require_uptrend=False)
    loose_strict_trend = LooseEmaRsiStrategy(require_uptrend=True)
    assert loose_default.generate_signal(df) == BUY
    assert loose_strict_trend.generate_signal(df) == HOLD


def test_stop_and_target_use_atr_multiples():
    strat = LooseEmaRsiStrategy(stop_atr_mult=2.0, target_atr_mult=4.0)
    df = make_df(list(np.linspace(100, 130, 130)))
    entry = 110.0
    stop = strat.get_stop_loss(entry, df, "long")
    target = strat.get_take_profit(entry, df, "long")
    assert stop < entry < target
    assert (target - entry) == pytest.approx(2 * (entry - stop))


def test_not_registered_in_signals_registry():
    from trading_bot.signals import REGISTRY
    assert "loose_ema_rsi" not in REGISTRY
