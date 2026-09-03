"""Regime detection tests — synthetic price series, no network access."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.regime import (  # noqa: E402
    BEAR,
    BULL,
    SIDEWAYS,
    detect_regime,
    strategy_allows_regime,
)


def make_df(closes) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": [c + 0.2 for c in closes], "low": [c - 0.2 for c in closes],
         "close": closes, "volume": [1000] * n},
        index=idx,
    )


def test_strong_uptrend_is_bull():
    df = make_df(list(np.linspace(100, 150, 80)))
    assert detect_regime(df) == BULL


def test_strong_downtrend_is_bear():
    df = make_df(list(np.linspace(150, 100, 80)))
    assert detect_regime(df) == BEAR


def test_choppy_flat_series_is_sideways():
    rng = np.random.RandomState(0)
    df = make_df(list(100 + rng.normal(0, 0.3, 80)))
    assert detect_regime(df) == SIDEWAYS


def test_too_few_bars_defaults_to_sideways():
    df = make_df([100.0] * 10)
    assert detect_regime(df) == SIDEWAYS


def test_low_adx_is_sideways_even_with_some_drift():
    # Very gentle drift shouldn't register as trending if ADX stays low.
    rng = np.random.RandomState(1)
    df = make_df(list(100 + np.linspace(0, 2, 80) + rng.normal(0, 0.3, 80)))
    result = detect_regime(df, trending_adx=40.0)  # force a high bar so this can't be trending
    assert result == SIDEWAYS


# ------------------------------------------------------------ strategy gating

def test_ema_rsi_only_allows_bull():
    assert strategy_allows_regime("ema_rsi", BULL) is True
    assert strategy_allows_regime("ema_rsi", BEAR) is False
    assert strategy_allows_regime("ema_rsi", SIDEWAYS) is False


def test_vwap_mean_reversion_only_allows_sideways():
    assert strategy_allows_regime("vwap_mean_reversion", SIDEWAYS) is True
    assert strategy_allows_regime("vwap_mean_reversion", BULL) is False


def test_breakout_strategies_allow_bull_and_bear_not_sideways():
    for name in ("bollinger_squeeze", "crypto_momentum"):
        assert strategy_allows_regime(name, BULL) is True
        assert strategy_allows_regime(name, BEAR) is True
        assert strategy_allows_regime(name, SIDEWAYS) is False


def test_unknown_strategy_is_never_gated():
    assert strategy_allows_regime("some_future_strategy", SIDEWAYS) is True
    assert strategy_allows_regime("some_future_strategy", BULL) is True


# --------------------------------------------------------------------------
# vectorised regime detection must be identical, not merely similar
# --------------------------------------------------------------------------

def _trending_df(n=300):
    import numpy as np
    idx = pd.date_range("2024-01-02", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(7)
    # A random walk with drift reversals, so all three regimes actually occur.
    steps = rng.normal(0, 1, n) + np.concatenate([
        np.full(n // 3, 0.8), np.full(n // 3, -0.8), np.full(n - 2 * (n // 3), 0.05)])
    close = 100 + np.cumsum(steps)
    return pd.DataFrame({"open": close, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": [1000] * n}, index=idx)


def test_series_matches_scalar_bar_for_bar():
    """The whole justification for the fast path: identical, not approximate."""
    from trading_bot.regime import detect_regime, detect_regime_series
    df = _trending_df()
    fast = detect_regime_series(df)
    for i in range(len(df)):
        assert fast[i] == detect_regime(df.iloc[: i + 1]), f"mismatch at bar {i}"


def test_series_covers_more_than_one_regime():
    """A test that only ever sees SIDEWAYS would pass while proving nothing."""
    from trading_bot.regime import detect_regime_series
    assert len(set(detect_regime_series(_trending_df()))) > 1


def test_series_handles_short_input():
    from trading_bot.regime import detect_regime_series
    assert detect_regime_series(_trending_df(5)) == ["SIDEWAYS"] * 5
