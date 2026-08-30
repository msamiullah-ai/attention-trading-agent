"""deploy_capital.py tests using fakes — no network access, no dependence on
whether any real symbol happens to be signaling BUY right now."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from trading_bot.broker import AccountSnapshot  # noqa: E402
from trading_bot.config import Config, ExecutionConfig, RiskConfig  # noqa: E402
from trading_bot.risk import RiskManager, TradeLog, TradeRecord  # noqa: E402

import deploy_capital as dc  # noqa: E402


def make_df(n=200):
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": [100.0] * n, "high": [100.5] * n, "low": [99.5] * n,
         "close": [100.0] * n, "volume": [1000] * n},
        index=idx,
    )


class FakeMarketData:
    def __init__(self, bars):
        self.bars = bars

    def get_bars(self, symbols, timeframe="1Min", lookback_days=30):
        return {s: self.bars[s] for s in symbols if s in self.bars}

    def get_last_price(self, symbol):
        return 100.0


def account():
    return AccountSnapshot(equity=100_000.0, cash=100_000.0, buying_power=100_000.0, last_equity=100_000.0, blocked=False)


def make_config(symbols):
    return Config(
        symbols=symbols, strategy="ema_rsi", strategy_params={}, timeframe="1Min",
        lookback_bars=250, risk=RiskConfig(), execution=ExecutionConfig(),
    )


def test_build_candidates_finds_nothing_when_no_signal_fires():
    config = make_config(["AAPL"])
    data = FakeMarketData({"AAPL": make_df()})  # flat prices -> no strategy fires BUY
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    candidates = dc.build_candidates(config, data, risk, account())
    assert candidates == []


def test_build_candidates_dedups_by_best_expectancy(monkeypatch):
    # Force every equity strategy to say BUY, so AAPL gets 3 candidates that
    # must be deduped down to 1 -- whichever strategy has the best track record.
    monkeypatch.setattr(
        dc, "EQUITY_STRATEGIES", ["ema_rsi", "vwap_mean_reversion", "bollinger_squeeze"]
    )
    for name in dc.EQUITY_STRATEGIES:
        cls = __import__("trading_bot.signals", fromlist=["REGISTRY"]).REGISTRY[name]
        monkeypatch.setattr(cls, "generate_signal", lambda self, df: "BUY")
        monkeypatch.setattr(cls, "get_stop_loss", lambda self, price, df, side: price - 5)
        monkeypatch.setattr(cls, "get_take_profit", lambda self, price, df, side: price + 10)

    config = make_config(["AAPL"])
    data = FakeMarketData({"AAPL": make_df()})
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))

    # Give vwap_mean_reversion the best track record so it should win the dedup.
    for _ in range(5):
        risk.record_trade(
            TradeRecord(
                symbol="AAPL", strategy="vwap_mean_reversion", side="long", qty=1,
                entry_price=100, exit_price=105, stop_loss=95,
                entry_time="t", exit_time="t", pnl=5, r_multiple=1.0,
            )
        )

    candidates = dc.build_candidates(config, data, risk, account())
    assert len(candidates) == 1
    assert candidates[0].symbol == "AAPL"
    assert candidates[0].strategy.name == "vwap_mean_reversion"


def test_build_candidates_skips_symbols_below_min_bars():
    config = make_config(["AAPL"])
    data = FakeMarketData({"AAPL": make_df(n=5)})  # far fewer than any strategy's min_bars
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    assert dc.build_candidates(config, data, risk, account()) == []


def test_candidate_weight_dollars_is_at_least_one():
    from trading_bot.signals import EmaRsiStrategy
    c = dc.Candidate("AAPL", EmaRsiStrategy(), qty=0.001, price=0.5, expectancy_r=0.0, stop=0.0, target=1.0)
    assert c.cost == pytest.approx(0.0005)
    assert c.weight_dollars == 1  # floored cost of $0.0005 still costs at least $1 in the DP
