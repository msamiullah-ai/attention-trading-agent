"""Read-only JSON API for the trading bot dashboard.

Run (from this directory, with the bot running in another terminal via
`python main.py`):

    pip install fastapi uvicorn
    uvicorn api:app --host 127.0.0.1 --port 8000

Endpoints
    GET /health                -> {"status": "ok", "mode": "PAPER"}
    GET /api/account           -> account snapshot
    GET /api/positions         -> open positions
    GET /api/signals           -> current BUY/SELL/HOLD per configured symbol
    GET /api/equity-curve      -> cumulative realized P/L over closed trades
    GET /api/risk-metrics      -> expectancy stats + Kelly sizing
    GET /api/options/positions -> held contracts: strike, expiry, DTE, collateral
    GET /api/options/cycles    -> the overlay's decision log, refusals included
    GET /api/options/config    -> the limits the overlay enforces

Design notes
    - Every route is read-only. This process never constructs a Trader and
      never submits orders.
    - The bot's per-cycle signal state lives in the bot's process, so this API
      recomputes signals from the latest bars itself for the configured symbols
      (same strategy classes, read-only).
    - JSON shapes and unit conventions live in api_types.py, the contract the
      frontend mirrors. All *_pct fields are percent, not fractions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from trading_bot.broker import AccountSnapshot, Broker, PositionSnapshot  # noqa: E402
from trading_bot.config import Config, Credentials, load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData, lookback_days_for  # noqa: E402
from trading_bot.risk import ExpectancyStats, RiskManager  # noqa: E402
from trading_bot.signals import Strategy, build_strategy  # noqa: E402

from trading_bot.options import parse_occ, days_to_expiry, moneyness, is_itm  # noqa: E402
from trading_bot.options_trader import load_options_config, read_reports  # noqa: E402

from trading_bot import allocation
from trading_bot.advisor_memory import AdvisorMemory
from trading_bot.config import load_agent_settings
from api_types import (  # noqa: E402
    AccountJson,
    EquityPointJson,
    HealthJson,
    PositionJson,
    RiskMetricsJson,
    OptionPlanJson,
    OptionPositionJson,
    OptionsConfigJson,
    SignalsJson,
    AdvisorVerdictJson,
    AdvisorStatsJson,
    AllocationJson,
)

app = FastAPI(title="alpaca-bot read-only API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- converters
# Pure functions from domain objects to API shapes. Each one pins the unit
# contract so the frontend can render values without any conversion of its own.


def account_to_json(account: AccountSnapshot, mode: str) -> AccountJson:
    return AccountJson(
        equity=round(account.equity, 2),
        cash=round(account.cash, 2),
        buying_power=round(account.buying_power, 2),
        daily_pnl_pct=round(account.daily_pnl_pct * 100, 4),
        mode=mode,
        blocked=account.blocked,
    )


def position_to_json(position: PositionSnapshot) -> PositionJson:
    return PositionJson(
        symbol=position.symbol,
        qty=round(position.qty, 4),
        market_value=round(position.market_value, 2),
        avg_entry_price=round(position.avg_entry_price, 4),
        unrealized_pl=round(position.unrealized_pl, 2),
        unrealized_plpc=round(position.unrealized_plpc * 100, 4),
        side=position.side,
    )


def pick_signal(strategy: Strategy, df) -> str:
    """Signal for one symbol from its bars; HOLD when there isn't enough history."""
    if df is None or len(df) < strategy.min_bars:
        return "HOLD"
    return strategy.generate_signal(df)


def stats_to_json(
    stats: ExpectancyStats,
    kelly_position_pct: float,
    strategy: str,
    window: int,
) -> RiskMetricsJson:
    return RiskMetricsJson(
        n=stats.n,
        win_rate=round(stats.win_rate * 100, 2),
        loss_rate=round(stats.loss_rate * 100, 2),
        expectancy=round(stats.expectancy_r, 3),
        breakeven_win_rate=round(stats.breakeven_win_rate * 100, 2),
        margin_of_safety=round(stats.margin_of_safety * 100, 2),
        kelly_position_pct=round(kelly_position_pct * 100, 2),
        strategy=strategy,
        window=window,
    )


def _mmdd(iso_stamp: str) -> str:
    try:
        return datetime.fromisoformat(iso_stamp).strftime("%m/%d")
    except ValueError:
        return iso_stamp[:5]


def trades_to_curve(risk: RiskManager) -> list[EquityPointJson]:
    """Cumulative realized P/L over closed trades, oldest first."""
    running = 0.0
    curve: list[EquityPointJson] = []
    for trade in sorted(risk.trade_log.trades, key=lambda t: t.exit_time):
        running += trade.pnl
        curve.append(EquityPointJson(date=_mmdd(trade.exit_time), equity=round(running, 2)))
    return curve


# ------------------------------------------------------------------- runtime
# Lazy singletons so boots with missing .env fail on first request, not import.


@dataclass
class Runtime:
    creds: Credentials
    cfg: Config
    broker: Broker
    data: MarketData
    strategy: Strategy

    def fresh_risk(self) -> RiskManager:
        """A RiskManager reading the trade log as it is right now.

        TradeLog loads state/trades.csv once in its constructor, so a cached
        RiskManager would pin the equity curve and expectancy stats to
        whatever had closed when this process served its first request. The
        bot appends to that CSV from its own process, so the API has to
        re-read it per request to see new trades.
        """
        return RiskManager(self.cfg.risk)


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        creds = load_credentials()
        cfg = load_config()
        _runtime = Runtime(
            creds=creds,
            cfg=cfg,
            broker=Broker(creds),
            data=MarketData(creds, feed=cfg.data.feed),
            strategy=build_strategy(cfg.strategy, cfg.strategy_params),
        )
    return _runtime


# ------------------------------------------------------------------- routes


@app.get("/health", response_model=HealthJson)
def health(rt: Runtime = Depends(get_runtime)) -> HealthJson:
    return HealthJson(status="ok", mode=rt.creds.mode)


@app.get("/api/account", response_model=AccountJson)
def account(rt: Runtime = Depends(get_runtime)) -> AccountJson:
    return account_to_json(rt.broker.get_account(), rt.creds.mode)


@app.get("/api/positions", response_model=list[PositionJson])
def positions(rt: Runtime = Depends(get_runtime)) -> list[PositionJson]:
    return [position_to_json(p) for p in rt.broker.get_positions().values()]


@app.get("/api/signals", response_model=SignalsJson)
def signals(rt: Runtime = Depends(get_runtime)) -> SignalsJson:
    lookback = lookback_days_for(rt.cfg.timeframe, rt.cfg.lookback_bars)
    bars = rt.data.get_bars(
        rt.cfg.symbols, timeframe=rt.cfg.timeframe, lookback_days=lookback
    )
    return {
        symbol: pick_signal(rt.strategy, bars.get(symbol))
        for symbol in rt.cfg.symbols
    }


@app.get("/api/equity-curve", response_model=list[EquityPointJson])
def equity_curve(rt: Runtime = Depends(get_runtime)) -> list[EquityPointJson]:
    return trades_to_curve(rt.fresh_risk())


@app.get("/api/risk-metrics", response_model=RiskMetricsJson)
def risk_metrics(rt: Runtime = Depends(get_runtime)) -> RiskMetricsJson:
    window = 50  # same window the terminal dashboard uses
    risk = rt.fresh_risk()
    stats = risk.expectancy(strategy=rt.cfg.strategy, window=window)
    return stats_to_json(
        stats,
        risk.kelly_position_pct(rt.cfg.strategy),
        rt.cfg.strategy,
        window,
    )


# --------------------------------------------------------------- options
# The overlay writes state/options_cycles.jsonl and these read it. The equity
# routes recompute signals from bars on each request, which cannot work here:
# an option chain fetch per symbol per request is far too slow to serve a
# dashboard, and it would put a second consumer on the same rate limit the bot
# is already using.


def option_position_to_json(symbol: str, position, spot: float | None) -> OptionPositionJson:
    """Shape one held contract for the UI.

    `collateral` is the field a share-shaped view has no room for and the one
    that matters: a short $92 put is a $9,200 obligation while its market value
    reads as $120.
    """
    occ = parse_occ(symbol)
    qty = position.qty
    short = qty < 0
    collateral = (occ.collateral(int(abs(qty)))
                  if (short and occ.right == "put") else 0.0)
    mny = moneyness(occ, spot) if spot else None
    return OptionPositionJson(
        symbol=symbol, underlying=occ.underlying, right=occ.right,
        strike=occ.strike, expiry=occ.expiry,
        dte=days_to_expiry(occ.expiry, datetime.now().date()),
        qty=round(qty, 4), side="short" if short else "long",
        market_value=round(position.market_value, 2),
        unrealized_pl=round(position.unrealized_pl, 2),
        collateral=round(collateral, 2),
        moneyness_pct=round(mny * 100, 2) if mny is not None else None,
        itm=is_itm(occ, spot) if spot else None,
    )


@app.get("/api/options/positions", response_model=list[OptionPositionJson])
def option_positions(rt: Runtime = Depends(get_runtime)) -> list[OptionPositionJson]:
    """Held option contracts.

    Identified by whether the symbol PARSES as OCC, not by an asset-class
    field, because the broker reports that inconsistently across endpoints and
    a share position mistaken for a contract would be shown against 100x the
    wrong collateral.
    """
    out: list[OptionPositionJson] = []
    for symbol, position in rt.broker.get_positions().items():
        occ = parse_occ(symbol)
        if occ is None:
            continue
        try:
            spot = rt.data.get_last_price(occ.underlying)
        except Exception:  # noqa: BLE001 - an unquoted name is unknown, not fatal
            spot = None
        out.append(option_position_to_json(symbol, position, spot))
    return sorted(out, key=lambda o: (o.moneyness_pct is None, o.moneyness_pct or 0))


@app.get("/api/options/cycles", response_model=list[OptionPlanJson])
def option_cycles(limit: int = 25) -> list[OptionPlanJson]:
    """The overlay's decision log, oldest first.

    Includes cycles where nothing traded, deliberately. Those are most of them,
    and they carry the reason -- "bearish signal with no shares; level 1 has no
    bearish option expression" -- which is the difference between a dashboard
    that shows what the agent did and one that shows what it decided.
    """
    return [OptionPlanJson(
        ts=r.get("ts", ""), summary=r.get("summary", ""),
        equity=r.get("equity", 0.0),
        collateral_posted=r.get("collateral_posted", 0.0),
        open_options=r.get("open_options", 0),
        plans=r.get("plans", []), orders=r.get("orders", []),
        skipped=r.get("skipped", []),
    ) for r in read_reports(limit=limit)]


@app.get("/api/advisor/verdicts", response_model=list[AdvisorVerdictJson])
def advisor_verdicts(limit: int = 30) -> list[AdvisorVerdictJson]:
    """The LLM's own review history, newest first.

    Read from the ledger the bot writes rather than recomputed: these are
    judgements already made, and re-asking the model what it thinks now would
    show something the bot never acted on.
    """
    mem = AdvisorMemory()
    rows = mem._rows[-limit:][::-1]
    return [AdvisorVerdictJson(
        ts=v.ts, symbol=v.symbol, side=v.side, action=v.action,
        reason=v.reason, price_at=v.price_at,
        move_pct=v.move_pct, correct=v.correct,
    ) for v in rows]


@app.get("/api/advisor/stats", response_model=AdvisorStatsJson)
def advisor_stats() -> AdvisorStatsJson:
    """Is the reviewer worth its latency? A number, not an opinion."""
    mem = AdvisorMemory()
    overall_rate, overall_n = mem.accuracy()
    veto_rate, veto_n = mem.accuracy("veto")
    settings = load_agent_settings()
    return AdvisorStatsJson(
        overall_rate=round(overall_rate, 4), overall_n=overall_n,
        veto_rate=round(veto_rate, 4), veto_n=veto_n,
        pending=sum(1 for v in mem._rows if v.correct is None),
        model=settings.model or "(none)",
    )


@app.get("/api/allocation", response_model=AllocationJson)
def current_allocation(rt: Runtime = Depends(get_runtime)) -> AllocationJson:
    """The equity/options split as it stands, recomputed from live account state.

    Recomputed rather than read from a log because it is a function of right
    now -- equity, options level, regime -- and a stale split shown as current
    is worse than none.
    """
    cfg = load_config()
    account = rt.broker.get_account()
    # options_level exists on MCPBroker only; the REST client has no endpoint
    # for it, and guessing a level is the one thing the options path refuses
    # to do anywhere else.
    level = (rt.broker.options_level()
             if hasattr(rt.broker, "options_level") else None)
    a = allocation.decide(account.equity, options_level=level)
    return AllocationJson(
        equity_budget=round(a.equity_budget, 2),
        options_budget=round(a.options_budget, 2),
        reserve=round(a.reserve, 2),
        regime=a.regime, reason=a.reason,
        enabled=cfg.execution.allocator == "dynamic",
    )


@app.get("/api/options/config", response_model=OptionsConfigJson)
def options_config() -> OptionsConfigJson:
    """The limits the overlay enforces, so the UI can render a cap rather than
    an unexplained refusal. max_collateral_pct is PERCENT, matching every other
    *_pct field in this contract."""
    c = load_options_config()
    return OptionsConfigJson(
        enabled=c.enabled, target_delta=c.target_delta,
        delta_min=c.delta_min, delta_max=c.delta_max,
        max_collateral_pct=round(c.max_collateral_pct * 100, 2),
        max_positions=c.max_positions,
        max_orders_per_day=c.max_orders_per_day,
        min_credit=c.min_credit,
    )


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415 - only needed when run directly

    uvicorn.run("api:app", host="127.0.0.1", port=8000)