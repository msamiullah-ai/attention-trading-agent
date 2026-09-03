"""Broker transport over Alpaca's MCP server, for stocks and options.

Why this exists and not just `broker.py`: the event requires Alpaca's Trading
API *via its MCP server or CLI*, and `broker.py` talks to `alpaca-py`'s
TradingClient directly. This is the same surface over the same account, reached
the way the rules ask for.

Deliberately a DROP-IN for `Broker`. Same method names, same `AccountSnapshot`
and `PositionSnapshot` returns, same synchronous calls -- so `trader.py`,
`api.py`, `run.py` and the existing tests do not change to use it.

The awkward part is that MCP is async and this codebase is not. Rather than
convert the bot, a single event loop runs on a daemon thread for the process
lifetime and every public method blocks on it. One loop, one connection, no
`asyncio.run()` per call (which would tear down the MCP session each time).

Three things about `place_option_order` that are not discoverable from the tool
schema, because the tool publishes no properties at all:

  1. `qty` and `limit_price` must be STRINGS. Numbers are rejected by pydantic
     validation with "Input should be a valid string".
  2. Unknown keyword arguments are rejected outright, so defensive aliases
     (`option_symbol` beside `symbol`, `quantity` beside `qty`) are themselves
     the failure. Canonical names only.
  3. A rejection is RETURNED as text rather than raised. A caller that only
     guards against exceptions will go on to poll for an order that was never
     created, and log nothing resembling an error.

All three are handled below.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .broker import AccountSnapshot, BrokerCapabilities, PositionSnapshot
from .data import is_crypto_symbol
from .config import Credentials
from .logger import get_logger

log = get_logger(__name__)

CALL_TIMEOUT = 60  # seconds; a wedged server must not block a cycle forever

# Logical name -> candidate tool names, most specific first. Resolved against
# what the server actually exposes, because tool names have changed between
# server versions and a hardcoded name is a silent zero-trade week.
TOOLS: dict[str, tuple[str, ...]] = {
    "account": ("get_account_info",),
    "clock": ("get_clock",),
    "positions": ("get_all_positions",),
    "position": ("get_open_position",),
    "orders": ("get_orders",),
    "cancel_order": ("cancel_order_by_id",),
    "cancel_all": ("cancel_all_orders",),
    "close_position": ("close_position",),
    "close_all": ("close_all_positions",),
    "place_stock": ("place_stock_order",),
    "place_option": ("place_option_order",),
    "option_chain": ("get_option_chain",),
    "option_contracts": ("get_option_contracts",),
    "option_snapshot": ("get_option_snapshot",),
    "asset": ("get_asset",),
    "stock_quote": ("get_stock_latest_quote",),
    "stock_bars": ("get_stock_bars",),
    # Retrieval. Optional: if the server does not expose these, `resolved` simply
    # has no entry and `retrieval.py` degrades to "unknown" rather than failing.
    "news": ("get_news",),
    "corporate_actions": ("get_corporate_action_announcements",),
}


def _unwrap(payload: Any) -> Any:
    """Strip the server's security envelope, if there is one.

    alpaca-mcp-server >= 3.x returns every tool result as
    `{"_alpaca_mcp_security": {...}, "data": <actual payload>}` rather than the
    payload itself. Read at the top level, every field a caller wants is simply
    absent -- so `get_account` reported equity 0.00 on a funded account and the
    options level came back UNKNOWN on an account approved for level 3. Nothing
    raised; the numbers were just quietly wrong, which is the worst way for a
    broker layer to fail.

    Unwrapped centrally so every tool benefits, and tolerant of the older shape
    so a server that does not use the envelope still works.

    The envelope's own `instructions` field is deliberately discarded rather
    than forwarded: it is text arriving from an external service, and this
    layer's job is to return data, not to relay directives into a prompt.
    """
    if isinstance(payload, dict) and "_alpaca_mcp_security" in payload and "data" in payload:
        payload = payload["data"]
    # A SECOND envelope, used by the list endpoints: {"result": [...]}. Missing
    # it meant `get_positions` returned {} while the account held four
    # positions -- so the bot re-bought symbols it already owned (DAN went to
    # 1.26 shares across two cycles), never hit max_open_positions, never
    # exited anything, and never watched a stop. Blind to its own book, with
    # nothing raised anywhere.
    #
    # Matched only when `result` is the SOLE key, so a genuine payload that
    # happens to carry a `result` field alongside others is left alone.
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


class MCPError(RuntimeError):
    """An MCP call failed, timed out, or returned a rejection instead of data."""


def _s(value: Any) -> str:
    """Numbers as strings, because this server's order tools demand it.

    `place_stock_order` and `place_option_order` both declare their numeric
    fields as strings and reject floats outright:

        Input should be a valid string [type=string_type, input_value=45.41]

    The module docstring already recorded this for options. It was not applied
    to `submit_order`, so the very first live equity order this bot ever
    attempted was rejected on four fields at once -- after the strategy, the
    scheduler, the risk gates and the LLM had all approved it. Every dry run
    passed, because a dry run stops one step before this call.

    Formatted without exponent notation: a fractional quantity like 0.0452
    must not reach the API as "4.52e-02".
    """
    if isinstance(value, float):
        return f"{value:.9f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MCPBroker(BrokerCapabilities):
    """Same surface as `Broker`, reached over the MCP server."""

    def __init__(self, creds: Credentials, server_args: list[str] | None = None):
        self.creds = creds
        # `uvx alpaca-mcp-server` is how the server is actually distributed --
        # it fetches and runs it in one step. Defaulting to the bare name meant
        # the launch died with a raw WinError 2 on any machine that had not
        # separately pip-installed it, which is most of them.
        # Override with MCP_SERVER_CMD (space separated) or the server_args arg.
        self._server_args = server_args or (
            os.environ.get("MCP_SERVER_CMD", "").split()
            or ["uvx", "alpaca-mcp-server"])
        self._init_capabilities()
        self.available: dict[str, Any] = {}
        self.resolved: dict[str, str] = {}
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start()

    # -- lifecycle -----------------------------------------------------------

    def _start(self) -> None:
        """Bring up the loop thread and connect. Blocks until tools resolve."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="mcp-broker")
        self._thread.start()
        if not self._ready.wait(timeout=120):
            raise MCPError("MCP server did not become ready within 120s")
        if self._error:
            raise MCPError(f"MCP startup failed: {self._error}")

    _error: str | None = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_idle())
        except Exception as exc:  # noqa: BLE001 - surfaced via self._error
            self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()

    async def _connect_and_idle(self) -> None:
        env = dict(os.environ)
        env["ALPACA_API_KEY"] = self.creds.api_key
        env["ALPACA_SECRET_KEY"] = self.creds.secret_key
        env["ALPACA_PAPER_TRADE"] = "True" if self.creds.paper else "False"

        params = StdioServerParameters(command=self._server_args[0],
                                       args=self._server_args[1:], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                listed = await session.list_tools()
                self.available = {t.name: t for t in listed.tools}
                for logical, names in TOOLS.items():
                    for n in names:
                        if n in self.available:
                            self.resolved[logical] = n
                            break
                missing = [k for k in ("account", "positions", "place_stock")
                           if k not in self.resolved]
                if missing:
                    self._error = f"server exposes no tool for: {missing}"
                log.info("MCP connected: %d tools exposed, %d resolved",
                         len(self.available), len(self.resolved))
                self._ready.set()
                # Hold the session open for the process lifetime.
                while True:
                    await asyncio.sleep(3600)

    # -- plumbing ------------------------------------------------------------

    def _call(self, logical: str, **kwargs) -> Any:
        """Synchronous wrapper around one MCP tool call."""
        name = self.resolved.get(logical)
        if name is None:
            raise MCPError(f"no server tool resolved for {logical!r}")
        if self._loop is None or self._session is None:
            raise MCPError("MCP session is not running")

        async def _invoke():
            return await asyncio.wait_for(
                self._session.call_tool(name, arguments=kwargs),
                timeout=CALL_TIMEOUT)

        future = asyncio.run_coroutine_threadsafe(_invoke(), self._loop)
        try:
            result = future.result(timeout=CALL_TIMEOUT + 10)
        except asyncio.TimeoutError as exc:
            raise MCPError(f"{name} timed out after {CALL_TIMEOUT}s") from exc

        payload = "\n".join(
            getattr(b, "text", "") or "" for b in result.content).strip()
        if getattr(result, "isError", False):
            raise MCPError(f"{name} failed: {payload[:400]}")
        if not payload:
            return None

        # A failed call can arrive as ordinary text with no error flag set. A
        # caller guarding only against exceptions would treat a rejection as
        # data, which is how an order that never existed gets polled for.
        low = payload.lstrip()[:200].lower()
        if low.startswith("error") or "validation error" in low:
            raise MCPError(f"{name} rejected: {payload[:400]}")

        try:
            return _unwrap(json.loads(payload))
        except json.JSONDecodeError:
            return payload


    def _fetch_shorting_enabled(self) -> bool:
        raw = self._call("account")
        return raw.get("shorting_enabled", False) if isinstance(raw, dict) else False

    def _fetch_fractionable(self, symbol: str) -> bool:
        # `symbol_or_asset_id` is the REST path parameter. `symbol` is dropped
        # silently and the server requests the unsubstituted path, which 404s.
        asset = self._call("asset", symbol_or_asset_id=symbol)
        return asset.get("fractionable", False) if isinstance(asset, dict) else False

    # -- account and clock ---------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        raw = self._call("account") or {}
        return AccountSnapshot(
            equity=_num(raw.get("equity")),
            cash=_num(raw.get("cash")),
            buying_power=_num(raw.get("buying_power")),
            last_equity=_num(raw.get("last_equity")) or _num(raw.get("equity")),
            blocked=bool(raw.get("trading_blocked") or raw.get("account_blocked")),
        )

    def options_level(self) -> int | None:
        """Approved options trading level, or None if the account will not say.

        Kept here rather than added to `AccountSnapshot` because that dataclass
        is shared with the equity broker, which has no use for it -- and this
        module is meant to be additive, not a change to the existing surface.

        None is returned rather than a default, and that distinction matters:
        assuming level 1 on an unknown account would silently block level 2
        strategies on an account that has it, while assuming higher would send
        orders that get rejected. Unknown is its own answer, and the caller
        refuses on it.
        """
        raw = self._call("account") or {}
        for key in ("options_trading_level", "options_approved_level",
                    "option_trading_level"):
            if raw.get(key) is not None:
                try:
                    return int(raw[key])
                except (TypeError, ValueError):
                    continue
        return None

    def is_market_open(self) -> bool:
        raw = self._call("clock") or {}
        return bool(raw.get("is_open"))

    def next_market_open(self) -> datetime:
        raw = self._call("clock") or {}
        nxt = raw.get("next_open")
        if not nxt:
            return datetime.now(timezone.utc)
        return datetime.fromisoformat(str(nxt).replace("Z", "+00:00"))

    # -- positions -----------------------------------------------------------

    def get_positions(self) -> dict[str, PositionSnapshot]:
        raw = self._call("positions") or []
        if isinstance(raw, dict):
            raw = raw.get("positions") or []
        out: dict[str, PositionSnapshot] = {}
        for p in raw:
            if not isinstance(p, dict):
                continue
            symbol = str(p.get("symbol") or "").upper()
            if not symbol:
                continue
            out[symbol] = PositionSnapshot(
                symbol=symbol,
                qty=_num(p.get("qty")),
                market_value=_num(p.get("market_value")),
                avg_entry_price=_num(p.get("avg_entry_price")),
                unrealized_pl=_num(p.get("unrealized_pl")),
                unrealized_plpc=_num(p.get("unrealized_plpc")),
                side=str(p.get("side") or "long").lower(),
            )
        return out

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        return self.get_positions().get(symbol.upper())

    # -- orders --------------------------------------------------------------

    def get_open_orders(self, symbol: str | None = None) -> list:
        raw = self._call("orders", status="open", limit=200) or []
        if isinstance(raw, dict):
            raw = raw.get("orders") or []
        if symbol:
            s = symbol.upper()
            raw = [o for o in raw if str(o.get("symbol", "")).upper() == s]
        return list(raw)

    def get_closed_orders(self, symbol: str, limit: int = 5) -> list:
        raw = self._call("orders", status="closed", limit=max(limit, 50)) or []
        if isinstance(raw, dict):
            raw = raw.get("orders") or []
        s = symbol.upper()
        return [o for o in raw if str(o.get("symbol", "")).upper() == s][:limit]

    def has_open_order(self, symbol: str) -> bool:
        return bool(self.get_open_orders(symbol))

    # -- placement -----------------------------------------------------------

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
        """Equity/crypto order. Same signature and bracket semantics as Broker.

        Crypto keeps `broker.py`'s behaviour exactly: Alpaca rejects advanced
        order classes for it, so a bracket request degrades to a plain GTC
        order and the caller watches stop/target itself.
        """
        symbol = symbol.upper()
        crypto = is_crypto_symbol(symbol)
        args: dict[str, Any] = {
            "symbol": symbol,
            "side": side.lower(),
            # Alpaca's canonical REST names. `quantity` and `order_type` read
            # naturally and are both rejected as unexpected keyword arguments --
            # the same trap the module docstring records for place_option_order,
            # hit again here because this payload was written separately.
            "qty": _s(qty),
            "type": order_type.lower(),
            "time_in_force": "gtc" if crypto else time_in_force.lower(),
        }
        if order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("limit_price is required for limit orders")
            args["limit_price"] = _s(round(limit_price, 2))

        # Alpaca rejects an advanced order_class on crypto AND on any
        # fractional quantity. `broker.py` already handles both; this payload
        # was written separately and only handled crypto, so every fractional
        # equity order sent from here carried a bracket and was refused --
        # silently, because the refusal comes back as data rather than raised.
        fractional = not crypto and float(qty) != int(float(qty))
        if (crypto or fractional) and (take_profit_price or stop_loss_price):
            log.warning("%s: Alpaca rejects bracket orders for %s; submitting "
                        "plain. Stop/target are watched in-process instead.",
                        symbol, "crypto" if crypto else f"fractional qty {qty}")
        elif take_profit_price or stop_loss_price:
            args["order_class"] = ("bracket" if (take_profit_price and stop_loss_price)
                                   else "oto")
            if take_profit_price:
                args["take_profit_limit_price"] = _s(round(take_profit_price, 2))
            if stop_loss_price:
                args["stop_loss_stop_price"] = _s(round(stop_loss_price, 2))

        result = self._call("place_stock", **args)
        # An accepted order always comes back with an id. Checking for one is
        # the only reliable rejection test here: this server returns refusals
        # as ordinary data with no error flag and no "error" prefix, so a
        # caller guarding on exceptions alone records a position that does not
        # exist and then polls five seconds for a fill that never comes.
        if isinstance(result, dict) and not result.get("id"):
            raise MCPError(f"place_stock_order created no order: {str(result)[:300]}")
        return result

    def submit_option_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "limit",
        limit_price: float | None = None,
        time_in_force: str = "day",
        position_intent: str | None = None,
    ):
        """Single-leg option order. Works for calls and puts, both sides.

        `symbol` is a full OCC contract symbol. `position_intent` matters to
        Alpaca -- selling to OPEN is not the same trade as buying to CLOSE --
        and is inferred from `side` when the caller does not say.

        qty and limit_price go as STRINGS: this tool validates with pydantic
        and rejects numbers outright, and it publishes no schema, so nothing
        type-checks them on the way out.
        """
        intent = position_intent or ("sell_to_open" if side.lower() == "sell"
                                     else "buy_to_open")
        args: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.lower(),
            "qty": str(int(qty)),
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower(),
            "position_intent": intent,
        }
        if order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("limit_price is required for limit orders")
            args["limit_price"] = f"{limit_price:.2f}"
        return self._call("place_option", **args)

    # -- convenience ---------------------------------------------------------

    def buy(self, symbol: str, qty: float, **kwargs):
        return self.submit_order(symbol, qty, "buy", **kwargs)

    def sell(self, symbol: str, qty: float, **kwargs):
        return self.submit_order(symbol, qty, "sell", **kwargs)

    def close_position(self, symbol: str):
        # `symbol_or_asset_id` is the REST path parameter. Passing `symbol`
        # does not fail loudly -- the server drops the unknown argument and
        # requests the unsubstituted path -- and this is the call `_maybe_exit`
        # and the circuit breaker both use, so the bot could open positions and
        # never close one. Fourth argument-name trap on this server; the
        # pattern is that a plausible name is silently answered wrong.
        return self._call("close_position", symbol_or_asset_id=symbol.upper())

    def close_all_positions(self, cancel_orders: bool = True):
        return self._call("close_all", cancel_orders=cancel_orders)

    def cancel_all_orders(self):
        return self._call("cancel_all")
