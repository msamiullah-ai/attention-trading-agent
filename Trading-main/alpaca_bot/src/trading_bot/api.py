"""Read-only HTTP API exposing the trading bot's live state to the React
dashboard in ../alpaca_bot/frontend.

Every route here is read-only: it only calls Broker.get_account() /
get_positions(), reads the on-disk trade log, and runs Strategy.generate_signal()
against the latest bars. None of that places an order -- only Trader.run_cycle()
(the actual trading loop in trader.py, run by main.py / `run.py trade`) does.
This process and the trading loop are independent and can run side by side;
they both read the same Alpaca account and the same state/trades.csv.

Run with `python serve.py` from the project root (see serve.py's docstring).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from .analytics import behavioral_stats
from .broker import Broker
from .config import Config, Credentials
from .correlation import compute_correlation_matrix
from .data import MarketData, is_crypto_symbol, lookback_days_for
from .knapsack import KnapsackItem, knapsack_select
from .regime import detect_regime, strategy_allows_regime
from .risk import ExpectancyStats, RiskManager, TradeLog
from .signals import BUY, REGISTRY, Strategy, build_strategy


def create_app(config: Config, creds: Credentials) -> FastAPI:
    app = FastAPI(title="Alpaca Trading Bot Dashboard API")

    broker = Broker(creds)
    data = MarketData(creds, feed=config.data.feed)
    risk = RiskManager(config.risk, trade_log=TradeLog())
    strategy: Strategy = build_strategy(config.strategy, config.strategy_params)

    @app.get("/api/account")
    def get_account() -> dict:
        try:
            account = broker.get_account()
        except Exception as exc:  # Alpaca/network error -> 502, don't crash the process
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "daily_pnl_pct": account.daily_pnl_pct,  # fraction, e.g. 0.0142 -- frontend multiplies by 100
            "blocked": account.blocked,
            # Not on broker.AccountSnapshot -- "PAPER"/"LIVE" lives on
            # Credentials instead, so it's merged in here.
            "mode": creds.mode,
        }

    @app.get("/api/positions")
    def get_positions() -> list[dict]:
        try:
            positions = broker.get_positions()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "market_value": p.market_value,
                "avg_entry_price": p.avg_entry_price,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_plpc": p.unrealized_plpc,
                "side": p.side,
            }
            for p in positions.values()
        ]

    @app.get("/api/signals")
    def get_signals() -> dict:
        """Current signal per configured symbol, computed from the latest
        bars. This calls the same Strategy.generate_signal() the live loop
        uses, just without acting on the result."""
        lookback_days = lookback_days_for(config.timeframe, config.lookback_bars)
        try:
            bars = data.get_bars(config.symbols, timeframe=config.timeframe, lookback_days=lookback_days)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        out: dict[str, str] = {}
        for symbol in config.symbols:
            df = bars.get(symbol)
            if df is None or len(df) < strategy.min_bars:
                continue
            out[symbol] = strategy.generate_signal(df)
        return out

    @app.get("/api/risk")
    def get_risk() -> dict | None:
        """Expectancy stats for the active strategy over its last 50 closed
        trades. Returns null (not fabricated zeros) if nothing has closed
        yet -- the frontend renders an honest "no data" state for that."""
        stats: ExpectancyStats = risk.expectancy(strategy=strategy.name, window=50)
        if stats.n == 0:
            return None

        return {
            "n": stats.n,
            "win_rate": stats.win_rate,
            "loss_rate": stats.loss_rate,
            "avg_win": stats.avg_win,
            "avg_loss": stats.avg_loss,
            "avg_r_winner": stats.avg_r_winner,
            "reward_risk_ratio": stats.reward_risk_ratio,
            "expectancy": stats.expectancy,  # dollars per trade
            "expectancy_r": stats.expectancy_r,  # R-multiples per trade
            "breakeven_win_rate": stats.breakeven_win_rate,
            "margin_of_safety": stats.margin_of_safety,
        }

    @app.get("/api/equity-curve")
    def get_equity_curve() -> list[dict]:
        """Reconstructs an equity series from the closed-trade log
        (state/trades.csv), anchored to the account's current equity.

        This is a reconstruction, NOT the broker's authoritative portfolio
        history -- deposits/withdrawals and open-position mark-to-market
        moves between trades aren't reflected, since nothing in this repo
        persists that. It's the most honest signal available from data
        this project actually logs. If Alpaca's portfolio-history endpoint
        gets wrapped in Broker later, swap this implementation for that.
        """
        try:
            account = broker.get_account()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        trades = sorted(risk.trade_log.trades, key=lambda t: t.exit_time)
        if not trades:
            today = datetime.now(timezone.utc).strftime("%m/%d")
            return [{"date": today, "equity": account.equity}]

        # Walk backwards from current equity, undoing each trade's P/L, to
        # find the equity level before the earliest trade in the window.
        running = account.equity
        for trade in reversed(trades):
            running -= trade.pnl
        starting_equity = running

        points = []
        running = starting_equity
        for trade in trades:
            running += trade.pnl
            date = trade.exit_time[:10] if len(trade.exit_time) >= 10 else trade.exit_time
            points.append({"date": date, "equity": round(running, 2)})
        return points

    return app
