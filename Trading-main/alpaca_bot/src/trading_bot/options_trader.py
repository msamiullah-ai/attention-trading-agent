"""The options cycle: direction from the equity strategy, expression in options.

A separate loop rather than an addition to `trader.py`, for two reasons. The
equity trader is built around bracket orders, share sizing and a per-symbol
stop/target that an option position does not have; threading options through it
would mean branching every step on asset class. And keeping them apart means
the equity path is untouched -- if this module is deleted, the bot behaves
exactly as it did before.

What it reuses, unchanged:

    MarketData   bars, exactly as the equity path fetches them
    Strategy     the SAME EMA/RSI, VWAP or Bollinger signal, via the registry
    MCPBroker    account, positions and orders over the MCP server

What it adds: the chain, the contract, the collateral arithmetic, and the
account-level check. The decision itself lives in `options_strategy.py` and is
pure, so what is here is I/O and bookkeeping.

RISK, and why it does not reuse RiskManager. That model sizes shares from a
Kelly fraction of notional and prices risk from a stop-loss distance. A
cash-secured put has neither: its exposure is `strike x 100` of posted
collateral, its premium is a few dollars, and there is no stop. Reusing it
would size a $22,600 obligation as though it were a $100 position. The limits
enforced here are the ones that actually bind on a short-premium book:

    max_collateral_pct    share of equity that may be posted against puts
    max_positions         concurrent option positions
    max_orders_per_day    a runaway detector, not a budget
    min_credit            below this the premium does not pay for the risk
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timezone
from typing import Any

from . import jsonl
from .config import Config, Credentials
from .data import MarketData
from .logger import get_logger
from .mcp_broker import MCPBroker, MCPError
from .options import OptionContract, parse_occ
from .options_strategy import OptionPlan, plan_option_trade
from .signals import build_strategy

log = get_logger(__name__)


@dataclass
class OptionsConfig:
    """Defaults chosen to be conservative rather than optimal, because nothing
    here has been backtested and a wrong number costs collateral, not points."""

    enabled: bool = True
    expiry: str | None = None          # None = nearest expiry the chain offers
    target_delta: float = 0.20
    delta_min: float = 0.15
    delta_max: float = 0.35
    max_collateral_pct: float = 0.50   # of equity, posted against short puts
    max_positions: int = 3
    max_contracts_per_order: int = 1
    max_orders_per_day: int = 10
    min_credit: float = 0.20           # per share, so $20 a contract
    limit_offset: float = 0.02         # price through the bid; mid does not fill


def load_options_config(path: "Path | None" = None) -> OptionsConfig:
    """Read options.yaml into an OptionsConfig, falling back to the defaults.

    Unknown keys are REJECTED rather than ignored. A silently dropped setting
    is the worst kind of config bug: `max_colateral_pct` with one L would leave
    the real cap at its default while the file says otherwise, and nothing
    would ever say so.
    """
    from pathlib import Path as _Path

    import yaml

    from .config import PROJECT_ROOT

    path = path or (PROJECT_ROOT / "options.yaml")
    if not _Path(path).exists():
        log.info("no options.yaml at %s; using built-in defaults", path)
        return OptionsConfig()

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"{path} must be a mapping, got {type(raw).__name__}")

    known = {f.name for f in fields(OptionsConfig)}
    unknown = set(raw) - known
    if unknown:
        raise RuntimeError(
            f"{path}: unknown option(s) {sorted(unknown)}. "
            f"Valid keys are {sorted(known)}")

    cfg = OptionsConfig(**raw)
    _validate(cfg, str(path))
    return cfg


def _validate(cfg: OptionsConfig, where: str) -> None:
    """Refuse a config that cannot mean what it says.

    Checked at load rather than at order time. Every knob here is one edit away
    from silently disarming a limit, and a disarmed limit raises no exception --
    it just stops enforcing.
    """
    problems: list[str] = []
    if not 0 < cfg.max_collateral_pct <= 1:
        problems.append(f"max_collateral_pct must be in (0, 1], got {cfg.max_collateral_pct}")
    if not 0 < cfg.delta_min <= cfg.delta_max < 1:
        problems.append(f"need 0 < delta_min <= delta_max < 1, got "
                        f"{cfg.delta_min}..{cfg.delta_max}")
    if not cfg.delta_min <= cfg.target_delta <= cfg.delta_max:
        problems.append(f"target_delta {cfg.target_delta} sits outside its own band "
                        f"{cfg.delta_min}..{cfg.delta_max}")
    if cfg.max_positions < 1:
        problems.append(f"max_positions must be >= 1, got {cfg.max_positions}")
    if cfg.max_contracts_per_order < 1:
        problems.append(f"max_contracts_per_order must be >= 1, got "
                        f"{cfg.max_contracts_per_order}")
    if cfg.max_orders_per_day < 1:
        problems.append(f"max_orders_per_day must be >= 1, got {cfg.max_orders_per_day}")
    if cfg.min_credit < 0:
        problems.append(f"min_credit cannot be negative, got {cfg.min_credit}")
    if cfg.limit_offset < 0:
        problems.append(f"limit_offset cannot be negative, got {cfg.limit_offset}")
    if problems:
        joined = "\n  - ".join(problems)
        raise RuntimeError(f"{where} is invalid:\n  - {joined}")


@dataclass
class CycleReport:
    """One cycle, in a form that logs and renders without further work."""

    plans: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    equity: float = 0.0
    collateral_posted: float = 0.0
    open_options: int = 0

    def to_dict(self) -> dict:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "summary": self.summary(),
            "equity": round(self.equity, 2),
            "collateral_posted": round(self.collateral_posted, 2),
            "open_options": self.open_options,
            "plans": self.plans,
            "orders": self.orders,
            "skipped": self.skipped,
        }

    def summary(self) -> str:
        if self.orders:
            return f"{len(self.orders)} option order(s) placed"
        if self.plans:
            return f"no order: {self.plans[-1].get('reason', 'unstated')}"
        return "no candidate produced a plan"


def append_report(report: "CycleReport", path: "Path | None" = None) -> None:
    """Append one cycle to state/options_cycles.jsonl.

    The API runs in a different process from the bot, so it cannot read the
    trader's memory. The equity side solves this by recomputing signals from
    bars; that will not work here because a chain fetch per symbol per request
    is far too expensive to serve a dashboard with.

    So the bot writes and the API reads. One line per cycle, append-only, which
    also means the decision log survives a restart -- including the cycles
    where nothing traded, which are most of them and are the ones that show the
    agent declining for stated reasons.

    Never raises. A bookkeeping failure must not end a trading cycle.
    """
    # `default=str` is applied here rather than in `jsonl.append`, because it
    # is a property of THIS payload -- a CycleReport carries dates and Decimals
    # -- not of JSONL writing. Letting it leak into the shared helper would
    # silently stringify a bug in some future caller instead of raising it.
    row = json.loads(json.dumps(report.to_dict(), default=str))
    jsonl.append(path or jsonl.state_path("options_cycles.jsonl"), row)


def read_reports(path: "Path | None" = None, limit: int = 50) -> list[dict]:
    """The most recent cycles, newest last. Empty when nothing has run yet."""
    return jsonl.read(path or jsonl.state_path("options_cycles.jsonl"), limit=limit)


class OptionsTrader:
    def __init__(self, cfg: Config, creds: Credentials, *,
                 broker: MCPBroker | None = None,
                 data: MarketData | None = None,
                 options: OptionsConfig | None = None,
                 collateral_budget: float | None = None):
        self.cfg = cfg
        self.opt = options or OptionsConfig()
        self.broker = broker or MCPBroker(creds)
        self.data = data or MarketData(creds, feed=cfg.data.feed)
        self.strategy = build_strategy(cfg.strategy, cfg.strategy_params)
        self.orders_today = 0
        self._session: date | None = None
        self._options_level: int | None = None
        # Ceiling on posted collateral from `allocation.decide`. None means
        # this side is running standalone against a share of full equity.
        self.collateral_budget = collateral_budget

    # -- helpers -------------------------------------------------------------

    def _roll_session(self, today: date) -> None:
        """The daily order cap resets with the session, not with the process."""
        if self._session != today:
            self._session = today
            self.orders_today = 0

    def open_option_positions(self) -> dict[str, OptionContract]:
        """Held option contracts, keyed by symbol.

        An option position is recognised by whether its symbol PARSES as OCC,
        not by an asset-class field. The broker reports the class
        inconsistently across endpoints, and a share position mistaken for a
        contract would be sized against the wrong collateral.
        """
        out: dict[str, OptionContract] = {}
        for symbol, pos in self.broker.get_positions().items():
            occ = parse_occ(symbol)
            if occ is None:
                continue
            out[symbol] = occ
        return out

    def posted_collateral(self) -> float:
        """Cash currently secured against SHORT puts.

        Long options post nothing and short calls are secured by shares, so
        only short puts consume the cash budget this method reports.
        """
        total = 0.0
        positions = self.broker.get_positions()
        for symbol, occ in self.open_option_positions().items():
            qty = positions[symbol].qty
            if qty < 0:                       # short only
                total += occ.collateral(int(abs(qty)))
        return total

    def shares_held(self, underlying: str) -> float:
        pos = self.broker.get_position(underlying.upper())
        return pos.qty if pos else 0.0

    def fetch_chain(self, underlying: str) -> list[OptionContract]:
        """Option chain for one underlying, normalised into OptionContract.

        Never raises: a chain that will not load is one name skipped, not a
        dead cycle. The reason is logged so a systematically missing chain is
        visible rather than silently absent.
        """
        try:
            raw = self.broker._call("option_chain", underlying_symbol=underlying)
        except Exception as exc:  # noqa: BLE001 - deliberately total
            # Total on purpose. The docstring promises one name is skipped
            # rather than the cycle dying, and catching only MCPError did not
            # deliver that: a parse error, a transport fault or a malformed
            # payload would all propagate and take the whole cycle with them.
            # A control that is narrower than its documentation is the failure
            # mode this codebase keeps finding.
            log.warning("chain unavailable for %s: %s", underlying, exc)
            return []
        rows = raw
        if isinstance(raw, dict):
            rows = (raw.get("option_contracts") or raw.get("contracts")
                    or raw.get("snapshots") or [])
        if not isinstance(rows, list):
            return []

        chain: list[OptionContract] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            occ = parse_occ(str(r.get("symbol") or ""))
            if occ is None:
                continue
            quote = r.get("latest_quote") or r.get("quote") or {}
            greeks = r.get("greeks") or {}
            chain.append(OptionContract(
                symbol=occ.symbol, underlying=occ.underlying, strike=occ.strike,
                expiry=occ.expiry, right=occ.right,
                bid=_f(quote.get("bid_price") or quote.get("bp")),
                ask=_f(quote.get("ask_price") or quote.get("ap")),
                delta=_f(greeks.get("delta")),
                implied_vol=_f(r.get("implied_volatility")),
                open_interest=_i(r.get("open_interest")),
            ))
        if self.opt.expiry:
            chain = [c for c in chain if c.expiry == self.opt.expiry]
        return chain

    # -- the cycle -----------------------------------------------------------

    def run_cycle(self, today: date | None = None) -> CycleReport:
        today = today or date.today()
        self._roll_session(today)
        report = CycleReport()

        if not self.opt.enabled:
            report.skipped.append({"reason": "options trading disabled in config"})
            append_report(report)
            return report

        account = self.broker.get_account()
        report.equity = account.equity
        if account.blocked:
            report.skipped.append({"reason": "account blocked for trading"})
            return report

        # Asked once per cycle, not once per symbol: it is an account
        # property, and an unknown level must refuse rather than default.
        self._options_level = self.broker.options_level()
        if self._options_level is None:
            report.skipped.append({
                "reason": "account options level unknown; refusing rather than "
                          "guessing a level the account may not have"})
            return report

        held = self.open_option_positions()
        report.open_options = len(held)
        report.collateral_posted = self.posted_collateral()

        if len(held) >= self.opt.max_positions:
            report.skipped.append({
                "reason": f"already holding {len(held)} option positions "
                          f"(max {self.opt.max_positions})"})
            return report

        # `collateral_budget` comes from `allocation.decide` when the two
        # sides are being coordinated; without it this is the standalone
        # behaviour, a share of full equity. Whichever is smaller wins, so
        # coordination can only ever tighten this side, never loosen it.
        ceiling = account.equity * self.opt.max_collateral_pct
        if self.collateral_budget is not None:
            ceiling = min(ceiling, self.collateral_budget)
        budget = ceiling - report.collateral_posted
        lookback = max(self.strategy.min_bars * 2, 60)
        bars = self.data.get_bars(self.cfg.symbols, timeframe=self.cfg.timeframe,
                                  lookback_days=lookback)

        for symbol in self.cfg.symbols:
            if self.orders_today >= self.opt.max_orders_per_day:
                report.skipped.append({
                    "reason": f"daily order cap {self.opt.max_orders_per_day} reached; "
                              "something is looping"})
                break

            df = bars.get(symbol)
            if df is None or len(df) < self.strategy.min_bars:
                continue

            signal = self.strategy.generate_signal(df)
            spot = float(df["close"].iloc[-1])
            chain = self.fetch_chain(symbol)

            plan = plan_option_trade(
                signal=signal, spot=spot, chain=chain,
                shares_held=self.shares_held(symbol),
                cash_available=max(0.0, min(account.cash, budget)),
                options_level=self._options_level,
                max_collateral=max(0.0, budget),
                max_contracts=self.opt.max_contracts_per_order,
                target_delta=self.opt.target_delta,
                delta_min=self.opt.delta_min, delta_max=self.opt.delta_max)

            row = plan.row() | {"symbol_underlying": symbol, "signal": signal}
            report.plans.append(row)
            if not plan.actionable:
                continue

            # The premium has to be worth the obligation. A contract paying
            # eight cents against 22,600 of posted collateral is not income,
            # it is a fee for accepting the risk.
            bid = plan.contract.bid or 0.0
            if bid < self.opt.min_credit:
                report.skipped.append({
                    "symbol": plan.contract.symbol,
                    "reason": f"credit {bid:.2f} below the {self.opt.min_credit:.2f} floor"})
                continue

            if self.broker.has_open_order(plan.contract.symbol):
                report.skipped.append({
                    "symbol": plan.contract.symbol,
                    "reason": "an order is already working on this contract"})
                continue

            # At or through the bid. A sell limit at mid does not fill on
            # paper, which is the most common cause of a week with no trades.
            limit = max(round(bid - self.opt.limit_offset, 2), 0.01)
            try:
                resp = self.broker.submit_option_order(
                    symbol=plan.contract.symbol, qty=plan.qty, side=plan.side,
                    order_type="limit", limit_price=limit)
            except MCPError as exc:
                report.skipped.append({"symbol": plan.contract.symbol,
                                       "reason": f"order rejected: {exc}"})
                continue

            self.orders_today += 1
            report.orders.append(row | {"limit": limit, "response": _brief(resp)})
            log.info("OPTION %s %s x%d @ %.2f -- %s", plan.side.upper(),
                     plan.contract.symbol, plan.qty, limit, plan.reason)

        append_report(report)
        return report


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _brief(resp: Any) -> Any:
    """Keep the order id and status; drop the rest so the log stays readable."""
    if isinstance(resp, dict):
        return {k: resp.get(k) for k in ("id", "client_order_id", "status", "qty")
                if k in resp}
    return str(resp)[:200] if resp is not None else None
