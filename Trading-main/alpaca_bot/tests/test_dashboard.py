"""Dashboard rendering tests — pure text output, no terminal needed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot import dashboard  # noqa: E402
from trading_bot.config import RiskConfig  # noqa: E402
from trading_bot.risk import RiskManager, TradeLog, TradeRecord  # noqa: E402

from test_trader import FakeBroker, FakeMarketData, FakeStrategy, make_config, make_creds, make_df  # noqa: E402
from trading_bot.trader import Trader  # noqa: E402


def make_trader_with_trades(n_trades=0, win=True):
    config = make_config()
    broker = FakeBroker()
    data = FakeMarketData()
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    for _ in range(n_trades):
        pnl = 100.0 if win else -100.0
        risk.record_trade(
            TradeRecord(
                symbol="AAPL", strategy="fake", side="long", qty=10,
                entry_price=100.0, exit_price=100.0 + pnl / 10, stop_loss=98.0,
                entry_time="2026-01-01T10:00:00", exit_time="2026-01-01T10:05:00",
                pnl=pnl, r_multiple=1.0 if win else -1.0,
            )
        )
    trader = Trader(config, make_creds(), broker=broker, data=data, risk=risk, strategy=FakeStrategy())
    return trader


def test_render_includes_header_and_sections():
    trader = make_trader_with_trades()
    text = dashboard.render(trader)
    assert "=== ALPACA AUTO-TRADER ===" in text
    assert "LIVE STATS" in text
    assert "Active Strategy: fake" in text
    assert "Daily Loss Limit:" in text


def test_render_shows_account_equity_and_positions():
    trader = make_trader_with_trades()
    text = dashboard.render(trader)
    assert "$100,000.00" in text
    assert "Open Positions: 0/" in text


def test_render_reflects_win_rate_from_trade_log():
    trader = make_trader_with_trades(n_trades=4, win=True)
    text = dashboard.render(trader)
    assert "Win Rate:       100.0%" in text
    assert "Expectancy:     +1.00R per trade" in text


def test_render_shows_no_signal_before_any_cycle():
    trader = make_trader_with_trades()
    text = dashboard.render(trader)
    assert "Last Signal: N/A - @ -" in text


def test_render_shows_last_signal_after_cycle():
    trader = make_trader_with_trades()
    trader.data.bars = {"AAPL": make_df()}
    trader.run_cycle()
    text = dashboard.render(trader)
    assert "Last Signal: HOLD AAPL @" in text


def test_kelly_label_reflects_multiplier():
    config = make_config(risk=RiskConfig(kelly_multiplier=0.25))
    broker = FakeBroker()
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    trader = Trader(config, make_creds(), broker=broker, data=FakeMarketData(), risk=risk, strategy=FakeStrategy())
    text = dashboard.render(trader)
    assert "Kelly (quarter):" in text


def test_daily_loss_used_pct_helper():
    assert dashboard._daily_loss_used_pct(-0.015, 0.03) == 0.5
    assert dashboard._daily_loss_used_pct(0.02, 0.03) == 0.0  # a gain, not a loss
    assert dashboard._daily_loss_used_pct(-0.10, 0.03) == 1.0  # clipped at 100%
