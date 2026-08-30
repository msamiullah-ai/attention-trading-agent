"""Terminal dashboard: a snapshot of account, live expectancy stats, and the
most recent signal. `render()` is pure (Trader in, text out) so it's testable
without a terminal; `print_dashboard()` is the side-effecting wrapper trader.py
calls once per cycle.
"""

from __future__ import annotations

from .risk import ExpectancyStats

_WIDTH = 41
_STATS_WINDOW = 50

_MULTIPLIER_LABELS = {0.25: "quarter", 0.5: "half", 1.0: "full"}


def _multiplier_label(multiplier: float) -> str:
    return _MULTIPLIER_LABELS.get(multiplier, f"{multiplier:.2f}x")


def _daily_loss_used_pct(daily_pnl_pct: float, max_daily_loss_pct: float) -> float:
    if max_daily_loss_pct <= 0:
        return 0.0
    used = max(0.0, -daily_pnl_pct) / max_daily_loss_pct
    return min(used, 1.0)


def render(trader) -> str:
    account = trader.broker.get_account()
    positions = trader.broker.get_positions()
    stats: ExpectancyStats = trader.risk.expectancy(strategy=trader.strategy.name, window=_STATS_WINDOW)
    kelly_pct = min(trader.risk.kelly_position_pct(trader.strategy.name), trader.cfg.risk.max_position_pct)
    kelly_label = f"Kelly ({_multiplier_label(trader.cfg.risk.kelly_multiplier)})"

    day_pnl_dollars = account.equity - account.last_equity
    loss_used = _daily_loss_used_pct(account.daily_pnl_pct, trader.cfg.risk.max_daily_loss_pct)

    last_signal = trader.last_signal.get(trader.last_signal_symbol, "N/A") if trader.last_signal_symbol else "N/A"
    last_symbol = trader.last_signal_symbol or "-"
    last_time = trader.last_signal_time or "-"

    lines = [
        "=== ALPACA AUTO-TRADER ===",
        f"Account: ${account.equity:,.2f} | Day P/L: ${day_pnl_dollars:+,.2f} ({account.daily_pnl_pct:+.2%})",
        f"Open Positions: {len(positions)}/{trader.cfg.risk.max_open_positions}",
        "-" * _WIDTH,
        f"LIVE STATS (last {_STATS_WINDOW} trades, {stats.n} logged):",
        f"  Win Rate:       {stats.win_rate:.1%}",
        f"  Avg Win (R):    {stats.avg_r_winner:.2f}R",
        "  Avg Loss (R):   1.00R",
        f"  Expectancy:     {stats.expectancy_r:+.2f}R per trade",
        f"  {kelly_label + ':':<16}{kelly_pct:.1%}",
        f"  Breakeven WR:   {stats.breakeven_win_rate:.1%}",
        f"  Margin of Safety: {stats.margin_of_safety:+.1%}",
        "-" * _WIDTH,
        f"Active Strategy: {trader.strategy.name}",
        f"Last Signal: {last_signal} {last_symbol} @ {last_time}",
        f"Daily Loss Limit: {loss_used:.1%} used",
    ]
    return "\n".join(lines)


def print_dashboard(trader) -> None:
    print(render(trader))
