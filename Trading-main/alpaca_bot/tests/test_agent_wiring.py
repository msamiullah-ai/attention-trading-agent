"""End-to-end checks that the advisor is actually wired into the trade path.

The unit tests in test_agent.py prove the advisor's logic. These prove the
wiring, which is where the real risk is: an advisor whose veto is computed and
then ignored passes every unit test in the file while doing nothing at all.
So each test here asserts on `broker.submitted_orders` -- what was actually
sent -- rather than on a returned object.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_trader import (  # noqa: E402
    FakeBroker, FakeMarketData, FakeStrategy, make_config, make_creds, make_df,
)
from trading_bot.risk import RiskManager, TradeLog  # noqa: E402
from trading_bot.signals import BUY  # noqa: E402
from trading_bot.trader import Trader  # noqa: E402


class CannedAdvisor:
    """Returns a fixed reply and records that it was asked."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        self.last_user = user
        return self.reply


def build(advisor=None, broker=None):
    config = make_config()
    broker = broker or FakeBroker()
    return broker, Trader(
        config, make_creds(), broker=broker,
        data=FakeMarketData(bars={"AAPL": make_df()}),
        risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
        strategy=FakeStrategy(signal=BUY), advisor=advisor,
    )


def test_no_advisor_trades_exactly_as_before(monkeypatch):
    """The default path must be untouched: no env var, no advisor, order sent."""
    monkeypatch.delenv("TRADING_BOT_ADVISOR", raising=False)
    broker, trader = build()
    assert trader.advisor is None
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1


def test_veto_prevents_the_order_being_sent():
    advisor = CannedAdvisor('{"action":"veto","reason":"earnings 2026-09-04"}')
    broker, trader = build(advisor=advisor)
    trader.run_cycle()
    assert advisor.calls == 1
    assert broker.submitted_orders == []


def test_shrink_reduces_the_quantity_actually_sent():
    advisor = CannedAdvisor('{"action":"shrink","size_factor":0.5,"reason":"vol"}')
    broker_plain, plain = build()
    plain.run_cycle()
    full_qty = broker_plain.submitted_orders[0]["qty"]

    broker, trader = build(advisor=advisor)
    trader.run_cycle()
    assert broker.submitted_orders[0]["qty"] == pytest.approx(int(full_qty * 0.5))


def test_confirm_cannot_enlarge_the_order():
    """The invariant that makes the whole design safe, checked at the wire."""
    broker_plain, plain = build()
    plain.run_cycle()
    full_qty = broker_plain.submitted_orders[0]["qty"]

    advisor = CannedAdvisor('{"action":"confirm","size_factor":10.0}')
    broker, trader = build(advisor=advisor)
    trader.run_cycle()
    assert broker.submitted_orders[0]["qty"] == full_qty


def test_advisor_outage_still_trades():
    """An API failure at a third party must not halt the bot."""
    class Broken:
        def complete(self, system, user):
            raise RuntimeError("503")

    broker, trader = build(advisor=Broken())
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1


def test_malformed_reply_still_trades():
    broker, trader = build(advisor=CannedAdvisor("I'd rather not say"))
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1


def test_retriever_is_absent_for_the_rest_broker():
    """FakeBroker has no `_call`, so retrieval must not be constructed."""
    _, trader = build(advisor=CannedAdvisor('{"action":"confirm"}'))
    assert trader.retriever is None


def test_recorded_trade_uses_the_shrunk_quantity():
    """A book that records the pre-shrink size would drift from the broker's."""
    advisor = CannedAdvisor('{"action":"shrink","size_factor":0.5,"reason":"v"}')
    broker, trader = build(advisor=advisor)
    trader.run_cycle()
    assert trader.open_trades["AAPL"].qty == broker.submitted_orders[0]["qty"]


def test_env_var_off_by_default(monkeypatch):
    monkeypatch.delenv("TRADING_BOT_ADVISOR", raising=False)
    _, trader = build()
    assert trader.advisor is None and trader.retriever is None


def test_every_advisor_outcome_is_logged(caplog):
    """A silent confirm is indistinguishable from an advisor that never ran."""
    import logging
    broker, trader = build(advisor=CannedAdvisor('{"action":"confirm"}'))
    with caplog.at_level(logging.INFO):
        trader.run_cycle()
    assert any("ADVISOR" in r.message or "ADVISOR" in r.getMessage()
               for r in caplog.records), "confirm produced no ADVISOR log line"


def test_veto_is_logged_too(caplog):
    import logging
    broker, trader = build(advisor=CannedAdvisor('{"action":"veto","reason":"halt"}'))
    with caplog.at_level(logging.INFO):
        trader.run_cycle()
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "ADVISOR" in text and "veto" in text.lower()


def test_a_short_is_refused_when_the_account_cannot_short():
    """A paper account defaults to shorting disabled. Discovering that from a
    403 at submission means the whole pipeline -- correlation, sizing, the LLM
    call -- was spent on a trade that was never placeable."""
    from trading_bot.signals import SELL
    broker = FakeBroker()
    broker.shorting = False
    config = make_config()
    trader = Trader(config, make_creds(), broker=broker,
                    data=FakeMarketData(bars={"AAPL": make_df()}),
                    risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
                    strategy=FakeStrategy(signal=SELL, allow_short=True))
    trader.run_cycle()
    assert broker.submitted_orders == []


def test_a_short_is_placed_when_the_account_can_short():
    from trading_bot.signals import SELL
    broker = FakeBroker()
    broker.shorting = True
    config = make_config()
    trader = Trader(config, make_creds(), broker=broker,
                    data=FakeMarketData(bars={"AAPL": make_df()}),
                    risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
                    strategy=FakeStrategy(signal=SELL, allow_short=True))
    trader.run_cycle()
    assert len(broker.submitted_orders) == 1


def test_a_rejected_order_does_not_end_the_cycle():
    """The broker can refuse for reasons no local check anticipates. Aborting
    would abandon every remaining candidate."""
    config = make_config()
    config.symbols = ["AAA", "BBB"]
    config.risk.correlation_threshold = 1.01

    class Refusing(FakeBroker):
        def submit_order(self, symbol, *a, **kw):
            if symbol == "AAA":
                raise RuntimeError("API rejected the order")
            return super().submit_order(symbol, *a, **kw)

    broker = Refusing()
    trader = Trader(config, make_creds(), broker=broker,
                    data=FakeMarketData(bars={s: make_df() for s in config.symbols}),
                    risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
                    strategy=FakeStrategy(signal=BUY))
    trader.run_cycle()                      # must not raise
    assert [o["symbol"] for o in broker.submitted_orders] == ["BBB"]


def test_a_rejected_order_releases_its_reserved_slot():
    """Otherwise a refused order permanently consumes one of max_open_positions
    for the rest of the cycle."""
    config = make_config()
    config.symbols = ["AAA", "BBB", "CCC"]
    config.risk.max_open_positions = 2
    config.risk.correlation_threshold = 1.01

    class RefuseFirst(FakeBroker):
        def submit_order(self, symbol, *a, **kw):
            if symbol == "AAA":
                raise RuntimeError("nope")
            return super().submit_order(symbol, *a, **kw)

    broker = RefuseFirst()
    trader = Trader(config, make_creds(), broker=broker,
                    data=FakeMarketData(bars={s: make_df() for s in config.symbols}),
                    risk=RiskManager(config.risk, trade_log=TradeLog(in_memory=True)),
                    strategy=FakeStrategy(signal=BUY))
    trader.run_cycle()
    assert len(broker.submitted_orders) == 2      # BBB and CCC both got in
