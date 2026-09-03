"""Tests for the REAL Broker.is_fractionable, not a fake.

Every other trader test injects FakeBroker, so nothing exercised this method on
the actual class -- and it shipped calling `self.trading`, an attribute that
does not exist, which crashed the live cycle the first time a BUY signal
reached sizing. A green suite proved nothing because the fake had its own
working implementation.

The Alpaca client is stubbed here rather than the whole broker, so the test
covers the constructor and the real method body.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.broker import Broker  # noqa: E402
from trading_bot.config import Credentials  # noqa: E402


class FakeAsset:
    def __init__(self, fractionable: bool):
        self.fractionable = fractionable


class FakeClient:
    def __init__(self, fractionable=True, raises=False):
        self._fractionable = fractionable
        self._raises = raises
        self.calls = 0

    def get_asset(self, symbol):
        self.calls += 1
        if self._raises:
            raise RuntimeError("alpaca down")
        return FakeAsset(self._fractionable)


def make_broker(client):
    with patch("trading_bot.broker.TradingClient", return_value=client):
        return Broker(Credentials(api_key="k", secret_key="s", paper=True))


def test_constructor_creates_the_cache():
    """The exact failure that reached production: the attribute was never set."""
    b = make_broker(FakeClient())
    assert hasattr(b, "_fractionable_cache")
    assert b._fractionable_cache == {}


def test_fractionable_symbol_reads_true():
    assert make_broker(FakeClient(fractionable=True)).is_fractionable("AAPL") is True


def test_non_fractionable_symbol_reads_false():
    assert make_broker(FakeClient(fractionable=False)).is_fractionable("BRK.A") is False


def test_result_is_cached_not_refetched():
    """Called once per candidate per cycle across ~1000 symbols; one API call
    each would dominate the cycle."""
    client = FakeClient()
    b = make_broker(client)
    for _ in range(5):
        b.is_fractionable("AAPL")
    assert client.calls == 1


def test_crypto_is_fractional_without_an_api_call():
    client = FakeClient()
    b = make_broker(client)
    assert b.is_fractionable("BTC/USD") is True
    assert client.calls == 0


def test_lookup_failure_is_conservative():
    """False costs a trade; True risks submitting an order Alpaca rejects."""
    assert make_broker(FakeClient(raises=True)).is_fractionable("AAPL") is False
