"""Trader tests using fake broker/data/strategy — no network access.

Trader accepts broker/data/risk/strategy as injectable dependencies
specifically so these tests can swap in fakes instead of hitting Alpaca.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.broker import AccountSnapshot, PositionSnapshot  # noqa: E402
from trading_bot.config import Config, Credentials, ExecutionConfig, RiskConfig  # noqa: E402
from trading_bot.risk import RiskManager, TradeLog  # noqa: E402
from trading_bot.signals import BUY, HOLD, SELL, Strategy  # noqa: E402
from trading_bot.trader import Trader  # noqa: E402


# ------------------------------------------------------------------- fakes

@dataclass
class FakeOrder:
    side: str
    filled_avg_price: float | None
    filled_at: datetime | None


class FakeBroker:
    def __init__(self, market_open: bool = True):
        self.market_open = market_open
        self.fractionable = True
        self.shorting = True
        self.positions: dict[str, PositionSnapshot] = {}
        self.closed_orders: dict[str, list[FakeOrder]] = {}
        self.submitted_orders: list[dict] = []
        self.closed_positions: list[str] = []
        self.cancelled_all = False
        self.closed_all = False
        self.open_order_symbols: set[str] = set()

    def is_market_open(self) -> bool:
        return self.market_open

    def shorting_enabled(self) -> bool:
        return self.shorting

    def is_fractionable(self, symbol: str) -> bool:
        # Default True so existing tests keep sizing the way they always have;
        # a test that cares about the non-fractionable path sets this False.
        return self.fractionable

    def next_market_open(self):
        return datetime(2026, 1, 1, tzinfo=timezone.utc)

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=100_000.0, cash=100_000.0, buying_power=100_000.0,
            last_equity=100_000.0, blocked=False,
        )

    def get_positions(self) -> dict[str, PositionSnapshot]:
        return dict(self.positions)

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        return self.positions.get(symbol)

    def has_open_order(self, symbol: str) -> bool:
        return symbol in self.open_order_symbols

    def submit_order(self, symbol, qty, side, **kwargs):
        self.submitted_orders.append({"symbol": symbol, "qty": qty, "side": side, **kwargs})
        # Simulate an immediate fill so _resolve_fill_price finds a position.
        fill_price = kwargs.get("_fill_price", 100.0)
        self.positions[symbol] = PositionSnapshot(
            symbol=symbol, qty=qty, market_value=qty * fill_price,
            avg_entry_price=fill_price, unrealized_pl=0.0, unrealized_plpc=0.0,
            side="long" if side == "buy" else "short",
        )
        return object()

    def close_position(self, symbol: str):
        self.closed_positions.append(symbol)
        self.positions.pop(symbol, None)

    def get_closed_orders(self, symbol: str, limit: int = 5) -> list:
        return self.closed_orders.get(symbol, [])

    def cancel_all_orders(self):
        self.cancelled_all = True

    def close_all_positions(self, cancel_orders: bool = True):
        self.closed_all = True


class FakeMarketData:
    def __init__(self, bars: dict[str, pd.DataFrame] | None = None, last_price: float = 100.0):
        self.bars = bars or {}
        self.last_price = last_price

    def get_bars(self, symbols, timeframe="1Min", lookback_days=30):
        return {s: self.bars[s] for s in symbols if s in self.bars}

    def get_last_price(self, symbol: str) -> float:
        return self.last_price


class FakeStrategy(Strategy):
    name = "fake"
    min_bars = 1

    def __init__(self, signal: str = HOLD, allow_short: bool = True, stop: float = 90.0, target: float = 110.0):
        self._signal = signal
        self.allow_short = allow_short
        self._stop = stop
        self._target = target

    def precompute(self, df):
        return {}

    def evaluate(self, df, ctx, i):
        return self._signal

    def generate_signal(self, df) -> str:
        return self._signal

    def get_stop_loss(self, entry_price, df, side):
        return self._stop

    def get_take_profit(self, entry_price, df, side):
        return self._target


def make_df(n=5):
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": [100.0] * n, "high": [100.5] * n, "low": [99.5] * n,
         "close": [100.0] * n, "volume": [1000] * n},
        index=idx,
    )


def make_config(**overrides) -> Config:
    defaults = dict(
        symbols=["AAPL"], strategy="fake", strategy_params={}, timeframe="1Min",
        lookback_bars=50, risk=RiskConfig(), execution=ExecutionConfig(require_market_open=True),
    )
    defaults.update(overrides)
    return Config(**defaults)


def make_creds() -> Credentials:
    return Credentials(api_key="k", secret_key="s", paper=True)


def make_trader(strategy, broker=None, data=None, risk=None, dry_run=False, tmp_path=None):
    config = make_config()
    broker = broker or FakeBroker()
    data = data or FakeMarketData(bars={"AAPL": make_df()})
    risk = risk or RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    return Trader(config, make_creds(), dry_run=dry_run, broker=broker, data=data, risk=risk, strategy=strategy)


# ------------------------------------------------------------------- tests

def test_run_cycle_skips_when_market_closed():
    broker = FakeBroker(market_open=False)
    trader = make_trader(FakeStrategy(signal=BUY), broker=broker)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_run_cycle_skips_when_risk_halted():
    broker = FakeBroker()
    config = make_config()
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    # Force the account gate closed via a blocked account snapshot.
    broker.get_account = lambda: AccountSnapshot(
        equity=100_000.0, cash=100_000.0, buying_power=100_000.0, last_equity=100_000.0, blocked=True,
    )
    trader = make_trader(FakeStrategy(signal=BUY), broker=broker, risk=risk)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_enters_long_position_on_buy_signal():
    broker = FakeBroker()
    trader = make_trader(FakeStrategy(signal=BUY, stop=90.0, target=110.0), broker=broker)
    trader.run_cycle()

    assert len(broker.submitted_orders) == 1
    order = broker.submitted_orders[0]
    assert order["side"] == "buy"
    assert order["stop_loss_price"] == 90.0
    assert order["take_profit_price"] == 110.0
    assert "AAPL" in trader.open_trades
    assert trader.open_trades["AAPL"].side == "long"


def test_long_only_strategy_does_not_open_short_on_sell():
    broker = FakeBroker()
    trader = make_trader(FakeStrategy(signal=SELL, allow_short=False), broker=broker)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_regime_gate_blocks_entry_for_known_strategy_in_wrong_regime():
    # Too few bars -> detect_regime() defaults to SIDEWAYS; ema_rsi only
    # trades BULL per regime.STRATEGY_REGIMES, so this must be blocked.
    strat = FakeStrategy(signal=BUY)
    strat.name = "ema_rsi"
    broker = FakeBroker()
    trader = make_trader(strat, broker=broker)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_regime_gate_does_not_block_strategies_outside_the_map():
    # "fake" isn't in STRATEGY_REGIMES -- unknown strategies are never gated.
    # (Covered implicitly by every other passing entry test too, but explicit here.)
    strat = FakeStrategy(signal=BUY)
    assert strat.name == "fake"
    broker = FakeBroker()
    trader = make_trader(strat, broker=broker)
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1


def test_correlation_gate_blocks_entry_when_hardcoded_pair_is_open():
    broker = FakeBroker()
    broker.positions["MSFT"] = PositionSnapshot(
        symbol="MSFT", qty=5, market_value=500.0, avg_entry_price=100.0,
        unrealized_pl=0.0, unrealized_plpc=0.0, side="long",
    )
    # AAPL/MSFT is a hardcoded pair in correlation.py -- blocked even though
    # MSFT's bars were never fetched this cycle (config.symbols is AAPL-only).
    trader = make_trader(FakeStrategy(signal=BUY), broker=broker)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_correlation_gate_allows_entry_when_open_position_is_unrelated():
    broker = FakeBroker()
    broker.positions["KO"] = PositionSnapshot(
        symbol="KO", qty=5, market_value=500.0, avg_entry_price=60.0,
        unrealized_pl=0.0, unrealized_plpc=0.0, side="long",
    )
    trader = make_trader(FakeStrategy(signal=BUY), broker=broker)
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1


def test_rejected_entry_does_not_submit_order():
    broker = FakeBroker()
    config = make_config()
    # max_open_positions=0 guarantees approve_entry rejects everything.
    risk = RiskManager(RiskConfig(max_open_positions=0), trade_log=TradeLog(in_memory=True))
    trader = make_trader(FakeStrategy(signal=BUY), broker=broker, risk=risk)
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_dry_run_does_not_submit_orders_or_track_trades():
    broker = FakeBroker()
    trader = make_trader(FakeStrategy(signal=BUY), broker=broker, dry_run=True)
    trader.run_cycle()
    assert broker.submitted_orders == []
    assert trader.open_trades == {}


def test_exit_signal_closes_long_position():
    broker = FakeBroker()
    broker.positions["AAPL"] = PositionSnapshot(
        symbol="AAPL", qty=5, market_value=500.0, avg_entry_price=100.0,
        unrealized_pl=0.0, unrealized_plpc=0.0, side="long",
    )
    trader = make_trader(FakeStrategy(signal=SELL), broker=broker)
    trader.run_cycle()
    assert broker.closed_positions == ["AAPL"]


def test_hold_signal_leaves_position_alone_for_bracket_to_manage():
    broker = FakeBroker()
    broker.positions["AAPL"] = PositionSnapshot(
        symbol="AAPL", qty=5, market_value=500.0, avg_entry_price=100.0,
        unrealized_pl=0.0, unrealized_plpc=0.0, side="long",
    )
    trader = make_trader(FakeStrategy(signal=HOLD), broker=broker)
    trader.run_cycle()
    assert broker.closed_positions == []


def test_finalize_trade_records_correct_pnl_for_long_win():
    from trading_bot.trader import OpenTrade

    broker = FakeBroker()
    broker.closed_orders["AAPL"] = [
        FakeOrder(side="sell", filled_avg_price=110.0, filled_at=datetime(2024, 1, 2, tzinfo=timezone.utc))
    ]
    trader = make_trader(FakeStrategy(), broker=broker)
    trader.open_trades["AAPL"] = OpenTrade(
        symbol="AAPL", strategy="fake", side="long", qty=10,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0, entry_time="2024-01-01T00:00:00",
    )
    # Position no longer open -> next cycle should reconcile it as closed.
    trader.run_cycle()

    trades = trader.risk.trade_log.trades
    assert len(trades) == 1
    assert trades[0].pnl == pytest.approx((110.0 - 100.0) * 10)
    assert trades[0].r_multiple == pytest.approx((110.0 - 100.0) / (100.0 - 95.0))
    assert "AAPL" not in trader.open_trades


def test_finalize_trade_falls_back_to_last_price_if_no_fill_found():
    from trading_bot.trader import OpenTrade

    broker = FakeBroker()  # no closed_orders entry for AAPL
    data = FakeMarketData(bars={"AAPL": make_df()}, last_price=95.0)
    trader = make_trader(FakeStrategy(), broker=broker, data=data)
    trader.open_trades["AAPL"] = OpenTrade(
        symbol="AAPL", strategy="fake", side="long", qty=10,
        entry_price=100.0, stop_loss=95.0, take_profit=110.0, entry_time="2024-01-01T00:00:00",
    )
    trader.run_cycle()

    trades = trader.risk.trade_log.trades
    assert len(trades) == 1
    assert trades[0].exit_price == pytest.approx(95.0)


def test_shutdown_closes_positions_when_live():
    broker = FakeBroker()
    trader = make_trader(FakeStrategy(), broker=broker, dry_run=False)
    trader.shutdown()
    assert broker.cancelled_all
    assert broker.closed_all


def test_shutdown_is_a_noop_in_dry_run():
    broker = FakeBroker()
    trader = make_trader(FakeStrategy(), broker=broker, dry_run=True)
    trader.shutdown()
    assert not broker.cancelled_all
    assert not broker.closed_all


# --------------------------------------------------------------- crypto (24/7)

def make_df_at_price(price: float, n=5):
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": [price] * n, "high": [price + 0.5] * n, "low": [price - 0.5] * n,
         "close": [price] * n, "volume": [1000] * n},
        index=idx,
    )


def test_crypto_not_blocked_when_equity_market_closed():
    broker = FakeBroker(market_open=False)
    config = make_config(symbols=["BTC/USD"])
    data = FakeMarketData(bars={"BTC/USD": make_df()})
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    trader = Trader(config, make_creds(), broker=broker, data=data, risk=risk, strategy=FakeStrategy(signal=BUY))
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1  # would have been skipped for an equity


def test_equity_still_blocked_when_market_closed_alongside_crypto():
    broker = FakeBroker(market_open=False)
    config = make_config(symbols=["AAPL", "BTC/USD"])
    data = FakeMarketData(bars={"AAPL": make_df(), "BTC/USD": make_df()})
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    trader = Trader(config, make_creds(), broker=broker, data=data, risk=risk, strategy=FakeStrategy(signal=BUY))
    trader.run_cycle()
    symbols_traded = {o["symbol"] for o in broker.submitted_orders}
    assert symbols_traded == {"BTC/USD"}


def test_crypto_protection_closes_on_stop_loss_breach():
    from trading_bot.trader import OpenTrade

    broker = FakeBroker()
    broker.positions["BTC/USD"] = PositionSnapshot(
        symbol="BTC/USD", qty=0.01, market_value=850.0, avg_entry_price=100_000.0,
        unrealized_pl=-150.0, unrealized_plpc=-0.015, side="long",
    )
    config = make_config(symbols=["BTC/USD"])
    data = FakeMarketData(bars={"BTC/USD": make_df_at_price(85_000.0)})  # price fell below stop
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    trader = Trader(config, make_creds(), broker=broker, data=data, risk=risk, strategy=FakeStrategy(signal=HOLD))
    trader.open_trades["BTC/USD"] = OpenTrade(
        symbol="BTC/USD", strategy="fake", side="long", qty=0.01,
        entry_price=100_000.0, stop_loss=90_000.0, take_profit=120_000.0,
        entry_time="2024-01-01T00:00:00",
    )
    trader.run_cycle()
    assert broker.closed_positions == ["BTC/USD"]


def test_crypto_protection_closes_on_take_profit_breach():
    from trading_bot.trader import OpenTrade

    broker = FakeBroker()
    broker.positions["BTC/USD"] = PositionSnapshot(
        symbol="BTC/USD", qty=0.01, market_value=1250.0, avg_entry_price=100_000.0,
        unrealized_pl=250.0, unrealized_plpc=0.025, side="long",
    )
    config = make_config(symbols=["BTC/USD"])
    data = FakeMarketData(bars={"BTC/USD": make_df_at_price(125_000.0)})  # price rose above target
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    trader = Trader(config, make_creds(), broker=broker, data=data, risk=risk, strategy=FakeStrategy(signal=HOLD))
    trader.open_trades["BTC/USD"] = OpenTrade(
        symbol="BTC/USD", strategy="fake", side="long", qty=0.01,
        entry_price=100_000.0, stop_loss=90_000.0, take_profit=120_000.0,
        entry_time="2024-01-01T00:00:00",
    )
    trader.run_cycle()
    assert broker.closed_positions == ["BTC/USD"]


def test_crypto_protection_does_nothing_within_range():
    from trading_bot.trader import OpenTrade

    broker = FakeBroker()
    broker.positions["BTC/USD"] = PositionSnapshot(
        symbol="BTC/USD", qty=0.01, market_value=1000.0, avg_entry_price=100_000.0,
        unrealized_pl=0.0, unrealized_plpc=0.0, side="long",
    )
    config = make_config(symbols=["BTC/USD"])
    data = FakeMarketData(bars={"BTC/USD": make_df_at_price(101_000.0)})  # between stop and target
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    trader = Trader(config, make_creds(), broker=broker, data=data, risk=risk, strategy=FakeStrategy(signal=HOLD))
    trader.open_trades["BTC/USD"] = OpenTrade(
        symbol="BTC/USD", strategy="fake", side="long", qty=0.01,
        entry_price=100_000.0, stop_loss=90_000.0, take_profit=120_000.0,
        entry_time="2024-01-01T00:00:00",
    )
    trader.run_cycle()
    assert broker.closed_positions == []


def test_crypto_protection_skips_positions_not_opened_by_bot():
    # A crypto position that exists but isn't in self.open_trades (e.g. opened
    # manually outside the bot) must not crash the protection check.
    broker = FakeBroker()
    broker.positions["BTC/USD"] = PositionSnapshot(
        symbol="BTC/USD", qty=0.01, market_value=1000.0, avg_entry_price=100_000.0,
        unrealized_pl=0.0, unrealized_plpc=0.0, side="long",
    )
    config = make_config(symbols=["BTC/USD"])
    data = FakeMarketData(bars={"BTC/USD": make_df_at_price(50_000.0)})  # would look like a stop breach
    risk = RiskManager(config.risk, trade_log=TradeLog(in_memory=True))
    trader = Trader(config, make_creds(), broker=broker, data=data, risk=risk, strategy=FakeStrategy(signal=HOLD))
    trader.run_cycle()
    assert broker.closed_positions == []
