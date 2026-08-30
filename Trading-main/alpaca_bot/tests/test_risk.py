"""Risk manager tests — expectancy math, Kelly sizing, and hard limits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.broker import AccountSnapshot, PositionSnapshot  # noqa: E402
from trading_bot.config import RiskConfig  # noqa: E402
from trading_bot.risk import (  # noqa: E402
    RiskManager,
    TradeLog,
    TradeRecord,
    compute_expectancy,
    kelly_fraction,
)


def account(equity=100_000.0, cash=100_000.0, buying_power=100_000.0, last_equity=100_000.0):
    return AccountSnapshot(
        equity=equity, cash=cash, buying_power=buying_power, last_equity=last_equity, blocked=False
    )


def position(symbol="AAPL", qty=10, market_value=2_000.0):
    return PositionSnapshot(
        symbol=symbol, qty=qty, market_value=market_value,
        avg_entry_price=market_value / qty, unrealized_pl=0.0, unrealized_plpc=0.0, side="long",
    )


def trade(pnl, r_multiple, strategy="ema_rsi", symbol="AAPL"):
    return TradeRecord(
        symbol=symbol, strategy=strategy, side="long", qty=10,
        entry_price=100.0, exit_price=100.0 + pnl / 10, stop_loss=98.0,
        entry_time="2026-01-01T10:00:00", exit_time="2026-01-01T10:05:00",
        pnl=pnl, r_multiple=r_multiple,
    )


# ------------------------------------------------------------- expectancy

def test_compute_expectancy_matches_manual_formula():
    trades = [
        trade(200.0, 2.0), trade(200.0, 2.0), trade(200.0, 2.0),   # 3 wins @ +2R
        trade(-100.0, -1.0), trade(-100.0, -1.0),                   # 2 losses @ -1R
    ]
    stats = compute_expectancy(trades)
    assert stats.n == 5
    assert stats.win_rate == pytest.approx(0.6)
    assert stats.loss_rate == pytest.approx(0.4)
    assert stats.avg_win == pytest.approx(200.0)
    assert stats.avg_loss == pytest.approx(100.0)
    assert stats.avg_r_winner == pytest.approx(2.0)
    # Expectancy = win_rate*avg_win - loss_rate*avg_loss = 0.6*200 - 0.4*100
    assert stats.expectancy == pytest.approx(0.6 * 200 - 0.4 * 100)
    # Expectancy_R = win_rate*avg_R_winner - loss_rate*1.0 = 0.6*2 - 0.4*1
    assert stats.expectancy_r == pytest.approx(0.6 * 2 - 0.4 * 1)


def test_breakeven_win_rate_formula():
    trades = [trade(300.0, 3.0), trade(-100.0, -1.0)]  # RR = 3
    stats = compute_expectancy(trades)
    assert stats.reward_risk_ratio == pytest.approx(3.0)
    assert stats.breakeven_win_rate == pytest.approx(1 / (1 + 3.0))
    assert stats.margin_of_safety == pytest.approx(stats.win_rate - stats.breakeven_win_rate)


def test_compute_expectancy_empty():
    stats = compute_expectancy([])
    assert stats.n == 0
    assert not stats.has_data


def test_compute_expectancy_all_wins_has_no_breakeven_denominator_issue():
    trades = [trade(100.0, 1.0), trade(100.0, 1.0)]
    stats = compute_expectancy(trades)
    assert stats.loss_rate == 0.0
    assert stats.avg_loss == 0.0


# ------------------------------------------------------------------ kelly

def test_kelly_fraction_formula():
    # kelly = (win_rate*RR - loss_rate) / RR
    assert kelly_fraction(0.6, 2.0) == pytest.approx((0.6 * 2.0 - 0.4) / 2.0)


def test_kelly_fraction_zero_when_no_edge_data():
    assert kelly_fraction(0.5, 0.0) == 0.0
    assert kelly_fraction(0.5, -1.0) == 0.0


def test_kelly_fraction_negative_when_losing_edge():
    # Win rate too low relative to RR -> negative Kelly (don't bet).
    assert kelly_fraction(0.2, 1.0) < 0


# --------------------------------------------------------------- trade log

def test_trade_log_persists_and_reloads(tmp_path):
    path = tmp_path / "trades.csv"
    log1 = TradeLog(path)
    log1.record(trade(150.0, 1.5))
    log1.record(trade(-80.0, -1.0))

    log2 = TradeLog(path)  # simulate a restart
    assert len(log2.trades) == 2
    assert log2.trades[0].pnl == pytest.approx(150.0)
    assert log2.trades[1].r_multiple == pytest.approx(-1.0)


def test_trade_log_filters_by_strategy(tmp_path):
    log = TradeLog(tmp_path / "trades.csv")
    log.record(trade(100.0, 1.0, strategy="ema_rsi"))
    log.record(trade(100.0, 1.0, strategy="vwap_mean_reversion"))
    assert len(log.for_strategy("ema_rsi")) == 1
    assert len(log.for_strategy("vwap_mean_reversion")) == 1


# -------------------------------------------------------------- RiskManager

@pytest.fixture
def rm(tmp_path):
    return RiskManager(
        RiskConfig(
            kelly_multiplier=0.5,
            max_position_pct=0.05,
            max_open_positions=3,
            max_daily_loss_pct=0.03,
            min_trades_for_kelly=20,
            fallback_position_pct=0.02,
            expectancy_window=20,
        ),
        trade_log=TradeLog(tmp_path / "trades.csv"),
    )


def test_sizing_uses_fallback_before_enough_history(rm):
    # No trades logged -> fallback_position_pct (2%) of equity / price.
    qty = rm.size_position(account(), price=200.0, strategy="ema_rsi", symbol="AAPL")
    assert qty == pytest.approx(10.0)  # 2% of 100k = 2000 / 200 = 10


def test_sizing_is_capped_by_max_position_pct_even_with_strong_edge(rm):
    for _ in range(20):
        rm.record_trade(trade(400.0, 4.0, strategy="ema_rsi"))  # huge edge, RR=4
    # Kelly would suggest a large fraction; hard cap still limits it to 5%.
    qty = rm.size_position(account(), price=200.0, strategy="ema_rsi", symbol="AAPL")
    assert qty <= (100_000 * 0.05) // 200


def test_sizing_is_capped_by_buying_power(rm):
    acct = account(equity=100_000.0, buying_power=500.0)
    qty = rm.size_position(acct, price=200.0, strategy="ema_rsi", symbol="AAPL")
    assert qty == pytest.approx(2.0)  # capped by buying power, not equity fraction


def test_sizing_is_fractional_for_crypto():
    rm_local = RiskManager(RiskConfig(fallback_position_pct=0.02), trade_log=TradeLog(in_memory=True))
    # BTC priced far above what a 2% equity slice would buy a whole unit of.
    qty = rm_local.size_position(account(), price=60_000.0, strategy="ema_rsi", symbol="BTC/USD")
    assert qty == pytest.approx(100_000 * 0.02 / 60_000, abs=1e-6)  # rounded to 6dp by design
    assert 0 < qty < 1  # fractional, not floored to 0


def test_daily_loss_limit_halts_trading(rm):
    losing = account(equity=96_000.0, last_equity=100_000.0)  # -4%
    decision = rm.check_account(losing)
    assert not decision.approved
    assert "daily loss" in decision.reason


def test_blocked_account_is_rejected(rm):
    acct = account()
    acct.blocked = True
    assert not rm.check_account(acct).approved


def test_expectancy_pause_skipped_before_window_filled(rm):
    for _ in range(5):
        rm.record_trade(trade(-100.0, -1.0, strategy="ema_rsi"))
    decision = rm.check_expectancy_pause("ema_rsi")
    assert decision.approved  # not enough trades yet to judge


def test_expectancy_pause_triggers_on_negative_expectancy(rm):
    for _ in range(20):
        rm.record_trade(trade(-100.0, -1.0, strategy="ema_rsi"))
    decision = rm.check_expectancy_pause("ema_rsi")
    assert not decision.approved
    assert "paused" in decision.reason


def test_expectancy_pause_does_not_block_other_strategies(rm):
    for _ in range(20):
        rm.record_trade(trade(-100.0, -1.0, strategy="ema_rsi"))
    decision = rm.check_expectancy_pause("vwap_mean_reversion")
    assert decision.approved


def test_no_duplicate_position(rm):
    decision = rm.approve_entry("AAPL", 200.0, "ema_rsi", account(), {"AAPL": position()})
    assert not decision.approved
    assert "already holding" in decision.reason


def test_max_open_positions_enforced(rm):
    positions = {s: position(s) for s in ("MSFT", "SPY", "TSLA")}
    decision = rm.approve_entry("AAPL", 200.0, "ema_rsi", account(), positions)
    assert not decision.approved
    assert "max open positions" in decision.reason


def test_approved_entry_returns_quantity(rm):
    decision = rm.approve_entry("AAPL", 200.0, "ema_rsi", account(), {})
    assert decision.approved
    assert decision.qty == pytest.approx(10.0)  # fallback sizing, no history yet


def test_rejects_when_share_unaffordable(rm):
    decision = rm.approve_entry("BRK.A", 700_000.0, "ema_rsi", account(), {})
    assert not decision.approved
    assert "below minimum" in decision.reason


def test_approve_entry_sizes_crypto_fractionally(rm):
    decision = rm.approve_entry("BTC/USD", 60_000.0, "ema_rsi", account(), {})
    assert decision.approved
    assert 0 < decision.qty < 1  # fractional, not rejected as "< 1"


def test_approve_entry_blocked_by_expectancy_pause(rm):
    for _ in range(20):
        rm.record_trade(trade(-100.0, -1.0, strategy="ema_rsi"))
    decision = rm.approve_entry("AAPL", 200.0, "ema_rsi", account(), {})
    assert not decision.approved
    assert "paused" in decision.reason
