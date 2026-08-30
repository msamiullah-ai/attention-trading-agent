"""Market data access: historical bars and latest prices.

Handles both equities (via StockHistoricalDataClient, gated by market hours)
and crypto (via CryptoHistoricalDataClient, trades 24/7, no feed/plan
restrictions). Callers pass a mixed symbol list -- a crypto pair like
"BTC/USD" is recognised by its "/" and routed to the crypto client
automatically, so trader.py/backtest.py don't need to know the difference.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestTradeRequest,
    StockBarsRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .config import Credentials
from .logger import get_logger

log = get_logger(__name__)


def is_crypto_symbol(symbol: str) -> bool:
    """Crypto pairs are written "BTC/USD"; equities never contain a slash."""
    return "/" in symbol

_TIMEFRAMES = {
    "1min": TimeFrame(1, TimeFrameUnit.Minute),
    "5min": TimeFrame(5, TimeFrameUnit.Minute),
    "15min": TimeFrame(15, TimeFrameUnit.Minute),
    "1hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1day": TimeFrame(1, TimeFrameUnit.Day),
}


_FEEDS = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}


def parse_timeframe(name: str) -> TimeFrame:
    key = name.strip().lower()
    if key not in _TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {name!r}. Use one of {list(_TIMEFRAMES)}")
    return _TIMEFRAMES[key]


# Approximate bars per calendar day, used to translate "N bars needed" into a
# lookback window wide enough to contain them (with headroom for weekends,
# holidays, and half days).
_BARS_PER_DAY = {
    "1min": 390,
    "5min": 78,
    "15min": 26,
    "1hour": 6.5,
    "1day": 1 / 1.6,  # ~1.6 calendar days per trading day
}


def _validate_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    before = len(df)
    df = df[df["volume"] > 0]
    if len(df) > 1:
        pct_change = df["close"].pct_change().abs()
        bad = pct_change > 0.25
        if bad.any():
            df = df[~bad]
    if len(df) < before:
        log.debug("Filtered %d bad bars from %s", before - len(df), symbol)
    return df


def lookback_days_for(timeframe: str, bars_needed: int, buffer_days: int = 5) -> int:
    """Calendar days to request so `bars_needed` bars are almost certainly present."""
    key = timeframe.strip().lower()
    if key not in _BARS_PER_DAY:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Use one of {list(_BARS_PER_DAY)}")
    per_day = _BARS_PER_DAY[key]
    return int(bars_needed / per_day) + buffer_days


class MarketData:
    """Historical bars and latest prices.

    `feed` must match your Alpaca data subscription: the free plan only permits
    "iex"; "sip" (full consolidated tape) requires a paid plan and returns
    "subscription does not permit querying recent SIP data" otherwise.
    """

    def __init__(self, creds: Credentials, feed: str = "iex"):
        # Data API uses the same keys regardless of paper/live.
        self._client = StockHistoricalDataClient(creds.api_key, creds.secret_key)
        self._crypto_client = CryptoHistoricalDataClient(creds.api_key, creds.secret_key)
        key = feed.strip().lower()
        if key not in _FEEDS:
            raise ValueError(f"Unsupported feed {feed!r}. Use one of {list(_FEEDS)}")
        self.feed = _FEEDS[key]

    def get_bars(
        self,
        symbols: list[str] | str,
        timeframe: str = "1Day",
        lookback_days: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """Return {symbol: DataFrame} indexed by timestamp with OHLCV columns."""
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [s.upper() for s in symbols]
        equities = [s for s in symbols if not is_crypto_symbol(s)]
        cryptos = [s for s in symbols if is_crypto_symbol(s)]

        out: dict[str, pd.DataFrame] = {}
        if equities:
            out.update(self._get_equity_bars(equities, timeframe, lookback_days))
        if cryptos:
            out.update(self._get_crypto_bars(cryptos, timeframe, lookback_days))
        return out

    def _get_equity_bars(self, symbols: list[str], timeframe: str, lookback_days: int) -> dict[str, pd.DataFrame]:
        end = datetime.now(timezone.utc) - timedelta(minutes=16)  # free-tier delay
        start = end - timedelta(days=lookback_days)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=parse_timeframe(timeframe),
            start=start,
            end=end,
            feed=self.feed,
        )
        df = self._client.get_stock_bars(request).df
        if df.empty:
            log.warning("No bar data returned for %s", symbols)
            return {}
        return self._split_by_symbol(df, symbols)

    def _get_crypto_bars(self, symbols: list[str], timeframe: str, lookback_days: int) -> dict[str, pd.DataFrame]:
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        start = end - timedelta(days=lookback_days)

        request = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=parse_timeframe(timeframe),
            start=start,
            end=end,
        )
        df = self._crypto_client.get_crypto_bars(request).df
        if df.empty:
            log.warning("No crypto bar data returned for %s", symbols)
            return {}
        return self._split_by_symbol(df, symbols)

    @staticmethod
    def _split_by_symbol(df: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol in df.index.get_level_values("symbol"):
                bar_df = df.xs(symbol, level="symbol").sort_index()
                bar_df = _validate_bars(bar_df, symbol)
                out[symbol] = bar_df
            else:
                log.warning("No bars for %s", symbol)
        return out

    def get_last_price(self, symbol: str) -> float | None:
        """Latest trade price, or None if unavailable."""
        symbol = symbol.upper()
        try:
            if is_crypto_symbol(symbol):
                resp = self._crypto_client.get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=symbol)
                )
            else:
                resp = self._client.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self.feed)
                )
            return float(resp[symbol].price)
        except Exception as exc:  # noqa: BLE001 - data gaps shouldn't kill the loop
            log.warning("Could not fetch last price for %s: %s", symbol, exc)
            return None
