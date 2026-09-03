"""Backtesting engine: replay a Strategy over historical bars, bar by bar.

Each bar's decision is made only from bars up to and including that bar (no
lookahead). Position sizing is Kelly-driven via the same RiskManager the live
trader uses, fed by the trade history the backtest itself accumulates as it
runs — so sizing starts flat (fallback_position_pct) and adapts exactly the
way it would live.

Not modelled: partial fills, multi-symbol portfolio margin, overnight gaps
beyond the bars provided. Slippage and commission are simple flat models
applied per fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .broker import AccountSnapshot
from .config import RiskConfig
from .data import is_crypto_symbol
from .risk import ExpectancyStats, RiskManager, TradeLog, TradeRecord, compute_expectancy, kelly_fraction
from .logger import get_logger
from .signals import BUY, SELL, Strategy

log = get_logger(__name__)

BARS_PER_YEAR_EQUITY = {
    "1min": 390 * 252,
    "5min": 78 * 252,
    "15min": 26 * 252,
    "1hour": 7 * 252,
    "1day": 252,
}

BARS_PER_YEAR_CRYPTO = {
    "1min": 525600,
    "5min": 105120,
    "15min": 35040,
    "1hour": 8760,
    "1day": 365,
}


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    starting_equity: float
    final_equity: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    trades: list[TradeRecord]
    expectancy: ExpectancyStats
    kelly_fraction: float
    equity_curve: pd.Series = field(repr=False)


def _bars_per_year(timeframe: str, is_crypto: bool = False) -> float:
    key = timeframe.strip().lower()
    table = BARS_PER_YEAR_CRYPTO if is_crypto else BARS_PER_YEAR_EQUITY
    if key not in table:
        raise ValueError(f"Unsupported timeframe {timeframe!r} for Sharpe annualisation")
    return table[key]


def _sharpe_ratio(equity: pd.Series, timeframe: str, is_crypto: bool = False) -> float:
    returns = equity.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(_bars_per_year(timeframe, is_crypto)))


def _max_drawdown_pct(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def resolve_starting_equity(explicit: float | None = None) -> float:
    """Whatever the account actually holds, unless told otherwise.

    Imported lazily so `backtest` keeps no import-time dependency on
    credentials -- a unit test that passes an explicit equity must never touch
    the network. Raises with an actionable message rather than substituting a
    default, because a wrong equity does not fail loudly, it quietly changes
    every sizing decision in the run.
    """
    if explicit is not None:
        return float(explicit)
    try:
        from .broker import Broker
        from .config import load_credentials
        equity = float(Broker(load_credentials()).get_account().equity)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "starting_equity was not given and the live account could not be "
            f"read ({exc}). Pass starting_equity=... explicitly."
        ) from exc
    if equity <= 0:
        raise RuntimeError(f"account equity is {equity}; nothing to backtest")
    log.info("backtest starting equity read from the live account: $%.2f", equity)
    return equity


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    symbol: str,
    risk_config: RiskConfig,
    timeframe: str = "1Day",
    starting_equity: float | None = None,
    slippage_pct: float = 0.0002,
    commission_per_trade: float = 0.0,
) -> BacktestResult:
    # No invented starting balance. A magic 10_000 default silently decides how
    # the whole run behaves: at that equity a 2% slice is $200, which cannot buy
    # one share of a $765 stock, so every entry is refused and the backtest
    # reports a clean, wrong "0 trades". Reading the real account instead means
    # the backtest models the account you actually have, whatever size it is.
    if starting_equity is None:
        starting_equity = resolve_starting_equity()

    if len(df) <= strategy.min_bars:
        raise ValueError(
            f"Need more than {strategy.min_bars} bars for {strategy.name}, got {len(df)}"
        )

    # In-memory only: Kelly sizing should adapt using this run's trades alone,
    # and a backtest must never read or write the live trade history.
    log = TradeLog(in_memory=True)
    risk = RiskManager(risk_config, trade_log=log)

    cash = starting_equity
    side: str | None = None       # None | "long" | "short"
    qty = 0.0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    entry_time = None

    equity_curve = pd.Series(index=df.index, dtype=float)

    def fake_account(current_cash: float) -> AccountSnapshot:
        return AccountSnapshot(
            equity=current_cash, cash=current_cash, buying_power=current_cash,
            last_equity=current_cash, blocked=False,
        )

    def fill_price(raw_price: float, is_buy_fill: bool) -> float:
        return raw_price * (1 + slippage_pct) if is_buy_fill else raw_price * (1 - slippage_pct)

    def close_position(exit_price: float, ts) -> None:
        nonlocal cash, side, qty, entry_price, stop_price, target_price, entry_time
        is_buy_fill = side == "short"  # covering a short is a buy
        fill = fill_price(exit_price, is_buy_fill)

        if side == "long":
            pnl = (fill - entry_price) * qty - commission_per_trade
            cash += qty * fill - commission_per_trade
            risk_per_share = entry_price - stop_price
        else:
            pnl = (entry_price - fill) * qty - commission_per_trade
            cash -= qty * fill + commission_per_trade
            risk_per_share = stop_price - entry_price

        r_multiple = pnl / (risk_per_share * qty) if risk_per_share > 0 else 0.0
        risk.record_trade(
            TradeRecord(
                symbol=symbol, strategy=strategy.name, side=side, qty=qty,
                entry_price=entry_price, exit_price=fill, stop_loss=stop_price,
                take_profit=target_price,
                entry_time=str(entry_time), exit_time=str(ts), pnl=pnl, r_multiple=r_multiple,
            )
        )
        side, qty, entry_price, stop_price, target_price, entry_time = None, 0.0, 0.0, 0.0, 0.0, None

    # Computed once over the full history (every indicator here is causal —
    # see the module docstring), then read bar-by-bar below. This is what
    # keeps a 6-month, 1-minute backtest (~70k bars) from being O(n^2): the
    # naive approach of calling generate_signal() on a growing window each
    # bar recomputes indicators over the entire history every single bar.
    ctx = strategy.precompute(df)

    for i in range(strategy.min_bars, len(df)):
        bar = df.iloc[i]
        ts = df.index[i]

        if side is not None:
            # Stop takes priority over target if both are touched intrabar.
            if side == "long" and bar["low"] <= stop_price:
                close_position(stop_price, ts)
            elif side == "short" and bar["high"] >= stop_price:
                close_position(stop_price, ts)
            elif side == "long" and bar["high"] >= target_price:
                close_position(target_price, ts)
            elif side == "short" and bar["low"] <= target_price:
                close_position(target_price, ts)
            else:
                signal = strategy.evaluate(df, ctx, i)
                if (side == "long" and signal == SELL) or (side == "short" and signal == BUY):
                    close_position(bar["close"], ts)

        if side is None:
            signal = strategy.evaluate(df, ctx, i)
            wants_long = signal == BUY
            wants_short = signal == SELL and getattr(strategy, "allow_short", True)

            if wants_long or wants_short:
                acct = fake_account(cash)
                price = bar["close"]
                new_side = "long" if wants_long else "short"
                sized_qty = risk.size_position(acct, price, strategy.name, symbol)

                min_qty = 1e-6 if is_crypto_symbol(symbol) else 1.0
                if sized_qty >= min_qty:
                    fill = fill_price(price, is_buy_fill=wants_long)
                    window = df.iloc[: i + 1]
                    stop = strategy.get_stop_loss(fill, window, new_side)
                    target = strategy.get_take_profit(fill, window, new_side)

                    side, qty, entry_price = new_side, sized_qty, fill
                    stop_price, target_price, entry_time = stop, target, ts
                    if wants_long:
                        cash -= qty * fill + commission_per_trade
                    else:
                        cash += qty * fill - commission_per_trade

        mark = df["close"].iloc[i]
        if side == "long":
            equity_curve.iloc[i] = cash + qty * mark
        elif side == "short":
            equity_curve.iloc[i] = cash - qty * mark
        else:
            equity_curve.iloc[i] = cash

    # Close any position still open at the end of the window at the last price.
    if side is not None:
        close_position(df["close"].iloc[-1], df.index[-1])
        equity_curve.iloc[-1] = cash

    equity_curve = equity_curve.iloc[strategy.min_bars :].ffill()
    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else starting_equity
    stats = compute_expectancy(log.trades)

    return BacktestResult(
        symbol=symbol,
        strategy=strategy.name,
        starting_equity=starting_equity,
        final_equity=final_equity,
        total_return_pct=(final_equity - starting_equity) / starting_equity,
        sharpe_ratio=_sharpe_ratio(equity_curve, timeframe, is_crypto=is_crypto_symbol(symbol)),
        max_drawdown_pct=_max_drawdown_pct(equity_curve),
        trades=log.trades,
        expectancy=stats,
        kelly_fraction=kelly_fraction(stats.win_rate, stats.reward_risk_ratio),
        equity_curve=equity_curve,
    )
