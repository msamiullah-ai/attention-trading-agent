"""Backtest engine tests using synthetic bars whose signals were verified via
manual probing (see test_signals.py note on Wilder RSI's recursive memory)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.backtest import run_backtest  # noqa: E402
from trading_bot.config import RiskConfig  # noqa: E402
from trading_bot.signals import EmaRsiStrategy, VwapMeanReversionStrategy  # noqa: E402


def make_df(closes, highs=None, lows=None, volumes=None, start="2024-01-02 09:30") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c + 0.3 for c in closes],
            "low": lows if lows is not None else [c - 0.3 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000] * n,
        },
        index=idx,
    )


def test_raises_when_not_enough_bars():
    strat = EmaRsiStrategy()
    df = make_df([100.0] * 5)
    with pytest.raises(ValueError, match="Need more than"):
        run_backtest(strat, df, "TEST", RiskConfig(), timeframe="1Min", starting_equity=10_000.0)


def test_long_only_strategy_never_opens_a_short():
    # Sustained rally holds RSI > 70 for many bars, which would repeatedly
    # emit SELL; a long-only strategy must never turn those into shorts.
    strat = EmaRsiStrategy()
    up = list(np.linspace(100, 102, 2))
    decline = list(np.linspace(up[-1], up[-1] - 15, 20))[1:]
    bounce = [decline[-1] + 10]
    rally = list(np.linspace(bounce[0], bounce[0] + 40, 30))
    df = make_df(up + decline + bounce + rally)

    result = run_backtest(strat, df, "TEST", RiskConfig(), timeframe="1Min", starting_equity=10_000.0)

    assert all(t.side == "long" for t in result.trades)
    # Exactly one round trip: one oversold-reversal entry, one overbought exit.
    assert len(result.trades) == 1


def test_ema_rsi_produces_expected_r_multiple_round_trip():
    strat = EmaRsiStrategy()
    up = list(np.linspace(100, 102, 2))
    decline = list(np.linspace(up[-1], up[-1] - 15, 20))[1:]
    bounce = [decline[-1] + 10]
    rally = list(np.linspace(bounce[0], bounce[0] + 40, 30))
    df = make_df(up + decline + bounce + rally)

    result = run_backtest(strat, df, "TEST", RiskConfig(), timeframe="1Min", starting_equity=10_000.0)

    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.pnl > 0
    assert trade.exit_price > trade.entry_price > trade.stop_loss
    assert result.final_equity == pytest.approx(
        result.starting_equity + trade.pnl, rel=1e-6
    )
    assert result.expectancy.n == 1
    assert result.expectancy.win_rate == pytest.approx(1.0)


def test_vwap_strategy_can_open_and_close_a_short():
    closes = (
        [100.0] * 20
        + [100.0, 100.0, 105.0]
        + [104, 103, 102, 101, 100.2, 100.0, 100.0]
    )
    df = make_df(closes, start="2024-01-02 16:00")  # UTC 16:00 -> ~11:00 America/New_York
    strat = VwapMeanReversionStrategy()

    result = run_backtest(strat, df, "TEST", RiskConfig(), timeframe="1Min", starting_equity=10_000.0)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short"
    assert trade.entry_price > trade.exit_price  # price fell back toward VWAP -> short profited
    assert trade.pnl > 0


def test_stop_loss_is_honoured_intrabar():
    # Price gaps straight through the stop on the very next bar after entry.
    strat = EmaRsiStrategy(stop_atr_mult=0.5, target_atr_mult=10.0)  # tight stop, far target
    up = list(np.linspace(100, 102, 2))
    decline = list(np.linspace(up[-1], up[-1] - 15, 20))[1:]
    bounce = [decline[-1] + 10]
    crash = [bounce[0] - 50]  # blows through any reasonable stop
    df = make_df(up + decline + bounce + crash)

    result = run_backtest(strat, df, "TEST", RiskConfig(), timeframe="1Min", starting_equity=10_000.0)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.pnl < 0
    # Exit fills at the stop price plus slippage (0.02% default), not exactly at it.
    assert trade.exit_price == pytest.approx(trade.stop_loss, rel=0.001)


def test_equity_curve_is_monotonic_length_and_starts_at_min_bars():
    strat = EmaRsiStrategy()
    df = make_df(list(np.linspace(100, 130, 60)))
    result = run_backtest(strat, df, "TEST", RiskConfig(), timeframe="1Min", starting_equity=10_000.0)
    assert len(result.equity_curve) == len(df) - strat.min_bars
    assert not result.equity_curve.isna().any()


def test_no_trades_still_returns_flat_result():
    strat = EmaRsiStrategy()
    df = make_df([100.0] * 60)  # flat prices, RSI/EMA never trigger anything
    result = run_backtest(strat, df, "TEST", RiskConfig(), timeframe="1Min", starting_equity=10_000.0)
    assert result.trades == []
    assert result.final_equity == pytest.approx(result.starting_equity)
    assert result.total_return_pct == pytest.approx(0.0)
