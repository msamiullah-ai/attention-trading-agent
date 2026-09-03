"""The options cycle, against a fake broker. No network, no credentials.

The fake mirrors the real payload SHAPES rather than accepting anything. A
fixture that agrees with whatever it is handed does not test the code, it
agrees with it -- and the bug that survives is always the one the fixture was
too permissive to catch.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.broker import AccountSnapshot, PositionSnapshot  # noqa: E402
from trading_bot.config import Config  # noqa: E402
from trading_bot.options_trader import OptionsConfig, OptionsTrader  # noqa: E402

TODAY = date(2026, 9, 1)
EXPIRY_CODE = "260904"


def occ(root: str, right: str, strike: float) -> str:
    return f"{root}{EXPIRY_CODE}{'P' if right == 'put' else 'C'}{int(strike*1000):08d}"


class FakeBroker:
    """Configurable. Defaults to a healthy account with an empty book."""

    def __init__(self, *, equity=100_000.0, cash=100_000.0, level=1,
                 positions=None, chain=None, open_orders=(), blocked=False,
                 reject=False):
        self.equity, self.cash, self.level = equity, cash, level
        self._positions = positions or {}
        self._chain = chain
        self._open_orders = set(open_orders)
        self.blocked = blocked
        self.reject = reject
        self.sent: list[dict] = []

    def get_account(self):
        return AccountSnapshot(equity=self.equity, cash=self.cash,
                               buying_power=self.equity * 2,
                               last_equity=self.equity, blocked=self.blocked)

    def options_level(self):
        return self.level

    def get_positions(self):
        return dict(self._positions)

    def get_position(self, symbol):
        return self._positions.get(symbol.upper())

    def has_open_order(self, symbol):
        return symbol in self._open_orders

    def _call(self, logical, **kw):
        assert logical == "option_chain"
        if self._chain is None:
            raise RuntimeError("no chain")
        return {"option_contracts": self._chain}

    def submit_option_order(self, **kw):
        if self.reject:
            from trading_bot.mcp_broker import MCPError
            raise MCPError("insufficient buying power")
        self.sent.append(kw)
        return {"id": "ord_1", "status": "accepted", "qty": kw.get("qty")}


class FakeData:
    """Bars that produce a chosen signal, without depending on real indicators."""

    def __init__(self, closes):
        self.closes = closes

    def get_bars(self, symbols, timeframe="1Min", lookback_days=200):
        idx = pd.date_range("2026-08-01", periods=len(self.closes), freq="D", tz="UTC")
        df = pd.DataFrame({"open": self.closes, "high": [c + 1 for c in self.closes],
                           "low": [c - 1 for c in self.closes], "close": self.closes,
                           "volume": [1_000_000] * len(self.closes)}, index=idx)
        return {s: df.copy() for s in (symbols if isinstance(symbols, list) else [symbols])}


class FixedStrategy:
    name, min_bars, allow_short = "fixed", 5, True

    def __init__(self, signal):
        self.signal = signal

    def generate_signal(self, df):
        return self.signal


def chain_rows(spot=100.0):
    rows = []
    for i, s in enumerate([98, 96, 94, 92]):
        rows.append({"symbol": occ("AAPL", "put", s),
                     "latest_quote": {"bid_price": 1.20, "ask_price": 1.26},
                     "greeks": {"delta": -(0.40 - 0.05 * i)}, "open_interest": 800})
    for i, s in enumerate([102, 104, 106, 108]):
        rows.append({"symbol": occ("AAPL", "call", s),
                     "latest_quote": {"bid_price": 1.10, "ask_price": 1.16},
                     "greeks": {"delta": 0.40 - 0.05 * i}, "open_interest": 800})
    return rows


def trader(signal="BUY", **broker_kw):
    cfg = Config(symbols=["AAPL"], strategy="ema_rsi", timeframe="1Day")
    broker_kw.setdefault("chain", chain_rows())
    t = OptionsTrader(cfg, creds=None, broker=FakeBroker(**broker_kw),
                      data=FakeData([100.0] * 40), options=OptionsConfig())
    t.strategy = FixedStrategy(signal)
    return t


# ------------------------------------------------------------------- the cycle


def test_bullish_signal_places_a_cash_secured_put():
    t = trader("BUY")
    rep = t.run_cycle(TODAY)
    assert len(rep.orders) == 1
    sent = t.broker.sent[0]
    assert sent["side"] == "sell" and "P00" in sent["symbol"]
    # priced THROUGH the bid -- a sell limit at mid does not fill on paper
    assert sent["limit_price"] < 1.20


def test_bearish_signal_with_no_shares_places_nothing():
    t = trader("SELL")
    rep = t.run_cycle(TODAY)
    assert not rep.orders and rep.plans
    assert "bearish" in rep.plans[-1]["reason"]


def test_bearish_signal_on_held_shares_writes_a_call():
    t = trader("SELL", positions={"AAPL": PositionSnapshot(
        "AAPL", 300, 30_000, 100, 0, 0, "long")})
    rep = t.run_cycle(TODAY)
    assert len(rep.orders) == 1
    assert "C00" in t.broker.sent[0]["symbol"]


# ------------------------------------------------------------------ the limits


def test_unknown_options_level_refuses_rather_than_guessing():
    t = trader("BUY", level=None)
    rep = t.run_cycle(TODAY)
    assert not rep.orders
    assert "level unknown" in rep.skipped[0]["reason"]


def test_blocked_account_stops_before_any_decision():
    t = trader("BUY", blocked=True)
    rep = t.run_cycle(TODAY)
    assert not rep.orders and not rep.plans


def test_position_cap_halts_the_cycle():
    held = {occ("XYZ", "put", 50.0): PositionSnapshot(
        occ("XYZ", "put", 50.0), -1, -100, 1, 0, 0, "short") for _ in range(1)}
    held.update({occ("ABC", "put", 40.0): PositionSnapshot(
        occ("ABC", "put", 40.0), -1, -100, 1, 0, 0, "short")})
    held.update({occ("DEF", "put", 30.0): PositionSnapshot(
        occ("DEF", "put", 30.0), -1, -100, 1, 0, 0, "short")})
    t = trader("BUY", positions=held)
    rep = t.run_cycle(TODAY)
    assert not rep.orders and "max 3" in rep.skipped[0]["reason"]


def test_collateral_already_posted_reduces_the_budget():
    # A short 92 put posts 9,200. Only short puts consume the cash budget.
    sym = occ("AAPL", "put", 92.0)
    t = trader("BUY", positions={sym: PositionSnapshot(sym, -1, -120, 1.2, 0, 0, "short")})
    rep = t.run_cycle(TODAY)
    assert rep.collateral_posted == 9_200.0


def test_a_long_option_posts_no_collateral():
    sym = occ("AAPL", "put", 92.0)
    t = trader("BUY", positions={sym: PositionSnapshot(sym, +1, 120, 1.2, 0, 0, "long")})
    assert t.posted_collateral() == 0.0


def test_credit_below_the_floor_is_refused():
    thin = [{"symbol": occ("AAPL", "put", 96.0),
             # spread kept tight so LIQUIDITY passes and the credit floor is
             # what actually refuses this -- otherwise the test would pass for
             # the wrong reason.
             "latest_quote": {"bid_price": 0.05, "ask_price": 0.06},
             "greeks": {"delta": -0.20}, "open_interest": 800}]
    t = trader("BUY", chain=thin)
    rep = t.run_cycle(TODAY)
    assert not rep.orders and "floor" in rep.skipped[-1]["reason"]


def test_an_existing_working_order_is_not_duplicated():
    t = trader("BUY")
    t.broker._open_orders = {occ("AAPL", "put", 92.0), occ("AAPL", "put", 94.0),
                             occ("AAPL", "put", 96.0), occ("AAPL", "put", 98.0)}
    rep = t.run_cycle(TODAY)
    assert not rep.orders and "already working" in rep.skipped[-1]["reason"]


def test_a_rejected_order_is_recorded_not_swallowed():
    t = trader("BUY", reject=True)
    rep = t.run_cycle(TODAY)
    assert not rep.orders
    assert "rejected" in rep.skipped[-1]["reason"]


def test_daily_order_cap_is_enforced():
    t = trader("BUY")
    t.orders_today = t.opt.max_orders_per_day
    t._session = TODAY
    rep = t.run_cycle(TODAY)
    assert not rep.orders and "looping" in rep.skipped[0]["reason"]


def test_the_cap_resets_on_a_new_session():
    t = trader("BUY")
    t.orders_today, t._session = 99, date(2026, 8, 31)
    t.run_cycle(TODAY)
    assert t.orders_today <= 1


def test_a_missing_chain_skips_the_name_rather_than_the_cycle():
    t = trader("BUY", chain=None)
    rep = t.run_cycle(TODAY)
    assert not rep.orders          # nothing traded
    assert rep.equity == 100_000   # but the cycle completed


def test_disabled_config_is_inert():
    cfg = Config(symbols=["AAPL"], strategy="ema_rsi", timeframe="1Day")
    t = OptionsTrader(cfg, creds=None, broker=FakeBroker(chain=chain_rows()),
                      data=FakeData([100.0] * 40),
                      options=OptionsConfig(enabled=False))
    t.strategy = FixedStrategy("BUY")
    rep = t.run_cycle(TODAY)
    assert not rep.orders and "disabled" in rep.skipped[0]["reason"]


def test_option_positions_are_identified_by_parsing_not_asset_class():
    # The broker reports asset class inconsistently across endpoints. A share
    # position mistaken for a contract would be sized against 100x the wrong
    # collateral.
    t = trader("BUY", positions={
        "AAPL": PositionSnapshot("AAPL", 100, 10_000, 100, 0, 0, "long"),
        occ("AAPL", "put", 92.0): PositionSnapshot(
            occ("AAPL", "put", 92.0), -1, -120, 1.2, 0, 0, "short")})
    held = t.open_option_positions()
    assert list(held) == [occ("AAPL", "put", 92.0)]


def test_every_skip_carries_a_reason():
    for kw in ({"level": None}, {"blocked": True}, {"reject": True}):
        rep = trader("BUY", **kw).run_cycle(TODAY)
        for s in rep.skipped:
            assert s.get("reason")
