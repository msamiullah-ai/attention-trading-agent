"""Thin wrapper around Alpaca's TradingClient.

Everything that talks to the broker goes through here, so the rest of the code
never touches the SDK directly and can be tested with a fake broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from alpaca.common.enums import Sort
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from .config import Credentials
from .data import is_crypto_symbol
from .logger import get_logger

log = get_logger(__name__)

_TIF = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "opg": TimeInForce.OPG}

# Alpaca is inconsistent about crypto symbol format across endpoints: orders
# use "BTC/USD" (matches submission and data requests), but positions come
# back as "BTCUSD" with no separator. We normalize everything to the slash
# form since that's what config.yaml/data.py/order submission all use.
_QUOTE_CURRENCIES = ("USDT", "USDC", "USD")


def _normalize_crypto_symbol(raw_symbol: str, asset_class) -> str:
    if asset_class != AssetClass.CRYPTO or "/" in raw_symbol:
        return raw_symbol
    for quote in _QUOTE_CURRENCIES:
        if raw_symbol.endswith(quote) and len(raw_symbol) > len(quote):
            return f"{raw_symbol[:-len(quote)]}/{quote}"
    return raw_symbol  # unrecognised quote currency -- leave as-is rather than guess


def _to_alpaca_symbol(symbol: str) -> str:
    if not is_crypto_symbol(symbol):
        return symbol
    base, quote = symbol.split("/", 1)
    return f"{base}{quote}"


@dataclass
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    last_equity: float
    blocked: bool

    @property
    def daily_pnl_pct(self) -> float:
        if self.last_equity <= 0:
            return 0.0
        return (self.equity - self.last_equity) / self.last_equity


@dataclass
class PositionSnapshot:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str


class BrokerCapabilities:
    """Cached yes/no facts about the account and its assets.

    Shared by `Broker` and `MCPBroker` because both need exactly this and both
    had their own copy -- which had already drifted: one guarded the payload
    with isinstance and the other did not, so the same lookup failure produced
    a different answer depending on the transport.

    Two rules hold for everything here, and they are the reason it is worth
    sharing rather than reimplementing:

      cached        these are properties of an account or an asset, not of the
                    market, and the entry path asks once per candidate per
                    cycle across a thousand symbols.
      False on
      failure       a lookup that errors costs a trade. Answering True would
                    submit an order the broker then rejects, which is worse:
                    the rejection arrives after the strategy, the scheduler,
                    the risk gates and the LLM have all spent work on it.

    Subclasses supply only the two fetches.
    """

    def _init_capabilities(self) -> None:
        self._fractionable_cache: dict[str, bool] = {}
        self._shorting_enabled: bool | None = None

    # -- supplied by the transport -------------------------------------------

    def _fetch_shorting_enabled(self) -> bool:
        raise NotImplementedError

    def _fetch_fractionable(self, symbol: str) -> bool:
        raise NotImplementedError

    # -- the shared behaviour ------------------------------------------------

    def shorting_enabled(self) -> bool:
        """Whether this account may open short positions.

        Asked rather than assumed: a paper account defaults to shorting
        disabled, and a strategy with allow_short=True will otherwise generate
        SELL signals all day that Alpaca refuses with a 403 at the last step --
        after the strategy, scheduler, risk gates and LLM have all spent work
        on them.
        """
        if self._shorting_enabled is None:
            try:
                self._shorting_enabled = bool(self._fetch_shorting_enabled())
            except Exception as exc:  # noqa: BLE001
                log.debug("shorting_enabled lookup failed: %s", exc)
                return False
        return self._shorting_enabled

    def is_fractionable(self, symbol: str) -> bool:
        """Whether Alpaca will accept a fractional quantity for this symbol."""
        if is_crypto_symbol(symbol):
            return True                     # crypto is always fractional
        cached = self._fractionable_cache.get(symbol)
        if cached is not None:
            return cached
        try:
            result = bool(self._fetch_fractionable(symbol))
        except Exception as exc:  # noqa: BLE001
            log.debug("fractionable lookup failed for %s: %s", symbol, exc)
            result = False
        self._fractionable_cache[symbol] = result
        return result


class Broker(BrokerCapabilities):
    """Buy/sell/inspect against Alpaca."""

    def _fetch_shorting_enabled(self) -> bool:
        return getattr(self._client.get_account(), "shorting_enabled", False)

    def _fetch_fractionable(self, symbol: str) -> bool:
        return getattr(self._client.get_asset(symbol), "fractionable", False)

    def __init__(self, creds: Credentials):
        self.paper = creds.paper
        self._init_capabilities()
        self._client = TradingClient(
            api_key=creds.api_key,
            secret_key=creds.secret_key,
            paper=creds.paper,
        )
        log.info("Broker initialised in %s mode", creds.mode)

    # ---------------------------------------------------------------- account

    def get_account(self) -> AccountSnapshot:
        a = self._client.get_account()
        return AccountSnapshot(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            last_equity=float(a.last_equity),
            blocked=bool(a.trading_blocked or a.account_blocked),
        )

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def next_market_open(self) -> datetime:
        return self._client.get_clock().next_open.astimezone(timezone.utc)

    # -------------------------------------------------------------- positions

    def get_positions(self) -> dict[str, PositionSnapshot]:
        out: dict[str, PositionSnapshot] = {}
        for p in self._client.get_all_positions():
            symbol = _normalize_crypto_symbol(p.symbol, p.asset_class)
            out[symbol] = PositionSnapshot(
                symbol=symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                side=str(p.side.value if hasattr(p.side, "value") else p.side),
            )
        return out

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        return self.get_positions().get(symbol.upper())

    # ----------------------------------------------------------------- orders

    def get_open_orders(self, symbol: str | None = None) -> list:
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol.upper()] if symbol else None,
        )
        return list(self._client.get_orders(filter=req))

    def get_closed_orders(self, symbol: str, limit: int = 5) -> list:
        """Most recent filled/closed orders for `symbol`, newest first."""
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol.upper()],
            limit=limit,
            direction=Sort.DESC,
        )
        return list(self._client.get_orders(filter=req))

    def has_open_order(self, symbol: str) -> bool:
        return bool(self.get_open_orders(symbol))

    def shorting_enabled(self) -> bool:
        """Whether this account may open short positions.

        Asked rather than assumed: a paper account defaults to shorting
        disabled, and a strategy with allow_short=True will then generate SELL
        signals all day that Alpaca refuses with a 403 at the very last step --
        after the strategy, scheduler, risk gates and LLM have all approved
        them. Cached because it is an account property, not a market one.
        """
        if self._shorting_enabled is None:
            try:
                self._shorting_enabled = bool(
                    getattr(self._client.get_account(), "shorting_enabled", False))
            except Exception as exc:  # noqa: BLE001
                log.debug("shorting_enabled lookup failed: %s", exc)
                return False        # conservative: cost a trade, not a rejection
        return self._shorting_enabled

    def is_fractionable(self, symbol: str) -> bool:
        """Whether Alpaca will accept a fractional quantity for this symbol.

        Cached for the process lifetime: the answer is a property of the asset,
        not of the market, and this is called once per candidate per cycle
        across a thousand symbols. Crypto is always fractional. On any lookup
        failure the answer is False, which costs a trade rather than risking a
        rejected order -- the conservative direction.
        """
        if is_crypto_symbol(symbol):
            return True
        cached = self._fractionable_cache.get(symbol)
        if cached is not None:
            return cached
        try:
            asset = self._client.get_asset(symbol)
            result = bool(getattr(asset, "fractionable", False))
        except Exception as exc:  # noqa: BLE001
            log.debug("fractionable lookup failed for %s: %s", symbol, exc)
            result = False
        self._fractionable_cache[symbol] = result
        return result

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ):
        """Submit an order. `side` is "buy" or "sell".

        If a take-profit or stop-loss price is given, the order is submitted as
        a bracket/OTO order so protection exists server-side even if the bot dies.

        Alpaca does not support bracket/OTO orders for crypto at all (rejected
        with "crypto orders not allowed for advanced order_class") -- for a
        crypto symbol this submits a plain order instead and the caller
        (trader.py) is responsible for watching price and closing manually.
        Crypto also requires GTC; "day" doesn't mean anything in a market
        that never closes.
        """
        symbol = symbol.upper()
        crypto = is_crypto_symbol(symbol)
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC if crypto else _TIF.get(time_in_force.lower(), TimeInForce.DAY)

        kwargs: dict = {
            "symbol": symbol,
            "qty": qty,
            "side": order_side,
            "time_in_force": tif,
        }

        # A fractional quantity carries the same restriction crypto does: Alpaca
        # rejects any advanced order_class on it, so the bracket cannot be
        # attached. Detected from the quantity itself rather than from an asset
        # lookup, because that is the thing Alpaca actually validates and it
        # costs no extra API call.
        fractional = not crypto and float(qty) != int(float(qty))
        if (crypto or fractional) and (take_profit_price or stop_loss_price):
            log.warning(
                "%s: Alpaca rejects bracket orders for %s; submitting a plain "
                "order. Stop/target are watched in-process instead.",
                symbol, "crypto" if crypto else f"fractional qty {qty}",
            )
        elif take_profit_price or stop_loss_price:
            # Bracket orders require GTC or DAY and cannot be used to close out
            # an existing position, only to open one.
            kwargs["order_class"] = "bracket" if (take_profit_price and stop_loss_price) else "oto"
            if take_profit_price:
                kwargs["take_profit"] = TakeProfitRequest(
                    limit_price=round(take_profit_price, 2)
                )
            if stop_loss_price:
                kwargs["stop_loss"] = StopLossRequest(stop_price=round(stop_loss_price, 2))

        if order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("limit_price is required for limit orders")
            request = LimitOrderRequest(limit_price=round(limit_price, 2), **kwargs)
        else:
            request = MarketOrderRequest(**kwargs)

        order = self._client.submit_order(order_data=request)
        log.info(
            "ORDER SUBMITTED %s %s %s qty=%s id=%s",
            order_type.upper(),
            side.upper(),
            symbol,
            qty,
            order.id,
        )
        return order

    def buy(self, symbol: str, qty: float, **kwargs):
        return self.submit_order(symbol, qty, "buy", **kwargs)

    def sell(self, symbol: str, qty: float, **kwargs):
        return self.submit_order(symbol, qty, "sell", **kwargs)

    def close_position(self, symbol: str):
        """Liquidate the whole position in `symbol`."""
        log.info("CLOSING position %s", symbol.upper())
        return self._client.close_position(_to_alpaca_symbol(symbol.upper()))

    def close_all_positions(self, cancel_orders: bool = True):
        log.warning("CLOSING ALL POSITIONS")
        return self._client.close_all_positions(cancel_orders=cancel_orders)

    def cancel_all_orders(self):
        log.info("Cancelling all open orders")
        return self._client.cancel_orders()
