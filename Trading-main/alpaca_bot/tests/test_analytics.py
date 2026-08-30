"""Behavioral analytics tests — no network access."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.analytics import behavioral_stats  # noqa: E402
from trading_bot.risk import TradeRecord  # noqa: E402


def trade(pnl, r_multiple, entry=100.0, stop=98.0, target=104.0):
    # planned R/R with entry=100, stop=98, target=104 -> reward=4, risk=2 -> 2.0
    return TradeRecord(
        symbol="AAPL", strategy="ema_rsi", side="long", qty=10,
        entry_price=entry, exit_price=entry + pnl / 10, stop_loss=stop, take_profit=target,
        entry_time="t0", exit_time="t1", pnl=pnl, r_multiple=r_multiple,
    )


def test_empty_trades_returns_none_stats():
    stats = behavioral_stats([])
    assert stats.n == 0
    assert stats.planned_win_r_avg is None
    assert stats.actual_win_r_avg is None
    assert stats.cutting_winners_short is False
    assert stats.letting_losers_run is False


def test_winners_matching_plan_are_not_flagged():
    # Planned R/R = 2.0, actual r_multiple = 2.0 -> realized exactly as planned.
    trades = [trade(200.0, 2.0), trade(200.0, 2.0)]
    stats = behavioral_stats(trades)
    assert stats.planned_win_r_avg == 2.0
    assert stats.actual_win_r_avg == 2.0
    assert stats.cutting_winners_short is False


def test_cutting_winners_short_is_flagged():
    # Planned R/R = 2.0 but only realizing 1.0R on average (50% of plan, below 70% threshold).
    trades = [trade(100.0, 1.0), trade(100.0, 1.0)]
    stats = behavioral_stats(trades)
    assert stats.planned_win_r_avg == 2.0
    assert stats.actual_win_r_avg == 1.0
    assert stats.cutting_winners_short is True


def test_letting_losers_run_is_flagged():
    # Losses averaging -1.5R, well past the planned 1R stop (threshold 1.3).
    trades = [trade(-150.0, -1.5), trade(-150.0, -1.5)]
    stats = behavioral_stats(trades)
    assert stats.actual_loss_r_avg == 1.5
    assert stats.letting_losers_run is True


def test_losses_at_planned_1r_are_not_flagged():
    trades = [trade(-100.0, -1.0), trade(-100.0, -1.0)]
    stats = behavioral_stats(trades)
    assert stats.actual_loss_r_avg == 1.0
    assert stats.letting_losers_run is False


def test_trades_without_recorded_target_are_excluded_from_win_r_comparison():
    # take_profit=0.0 -> planned_reward_risk is None -> excluded from the
    # planned-vs-actual comparison, but still counts toward total n.
    no_plan = TradeRecord(
        symbol="AAPL", strategy="ema_rsi", side="long", qty=10,
        entry_price=100.0, exit_price=105.0, stop_loss=98.0, take_profit=0.0,
        entry_time="t0", exit_time="t1", pnl=50.0, r_multiple=2.5,
    )
    stats = behavioral_stats([no_plan])
    assert stats.n == 1
    assert stats.planned_win_r_avg is None
    assert stats.actual_win_r_avg is None


def test_summary_lines_mentions_warning_when_flagged():
    trades = [trade(100.0, 1.0), trade(100.0, 1.0)]  # cutting winners short
    stats = behavioral_stats(trades)
    lines = "\n".join(stats.summary_lines())
    assert "cutting winners short" in lines.lower()


def test_summary_lines_handles_no_data_gracefully():
    stats = behavioral_stats([])
    lines = stats.summary_lines()
    assert len(lines) == 1
    assert "not enough" in lines[0].lower()
