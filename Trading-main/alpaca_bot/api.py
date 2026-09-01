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

from api_types import (  # noqa: E402
    AccountJson,
    EquityPointJson,
    HealthJson,
    PositionJson,
    RiskMetricsJson,
    SignalsJson,
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
    risk: RiskManager


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
            risk=RiskManager(cfg.risk),
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
    return trades_to_curve(rt.risk)


@app.get("/api/risk-metrics", response_model=RiskMetricsJson)
def risk_metrics(rt: Runtime = Depends(get_runtime)) -> RiskMetricsJson:
    window = 50  # same window the terminal dashboard uses
    stats = rt.risk.expectancy(strategy=rt.cfg.strategy, window=window)
    return stats_to_json(
        stats,
        rt.risk.kelly_position_pct(rt.cfg.strategy),
        rt.cfg.strategy,
        window,
    )


if __name__ == "__main__":
    import uvicorn  # noqa: PLC0415 - only needed when run directly

    uvicorn.run("api:app", host="127.0.0.1", port=8000)