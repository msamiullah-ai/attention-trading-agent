"""Run the configured strategy through the backtest engine against real
historical bars pulled from Alpaca, and save a CSV trade log + equity curve
PNG per symbol.

    python scripts/backtest.py --days 180
    python scripts/backtest.py --symbols AAPL,MSFT --strategy vwap_mean_reversion
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt  # noqa: E402

from trading_bot.backtest import run_backtest  # noqa: E402
from trading_bot.config import load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData, lookback_days_for  # noqa: E402
from trading_bot.logger import setup_logging  # noqa: E402
from trading_bot.risk import CSV_FIELDS  # noqa: E402
from trading_bot.signals import build_strategy  # noqa: E402

OUTPUT_DIR = ROOT / "backtests"


def _filename_safe(symbol: str) -> str:
    """Crypto pairs like "BTC/USD" contain a slash, which a filesystem reads
    as a path separator -- swap it for a dash so the file lands in
    backtests/ instead of failing with "No such directory: backtests/BTC"."""
    return symbol.replace("/", "-")


def save_trade_log(symbol: str, strategy_name: str, trades) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{strategy_name}_{_filename_safe(symbol)}_trades.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in trades:
            writer.writerow(t.to_csv_row())
    return path


def save_equity_curve(symbol: str, strategy_name: str, equity_curve) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{strategy_name}_{_filename_safe(symbol)}_equity.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve.index, equity_curve.values, linewidth=1.2)
    ax.set_title(f"{strategy_name} — {symbol} equity curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest a strategy against real Alpaca history")
    parser.add_argument("--days", type=int, default=180, help="Calendar days of history")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Starting equity per symbol")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated override of config.yaml symbols")
    parser.add_argument("--strategy", type=str, default=None, help="Override config.yaml strategy name")
    args = parser.parse_args()

    setup_logging("WARNING")
    config = load_config()
    creds = load_credentials()

    strategy_name = args.strategy or config.strategy
    strategy = build_strategy(strategy_name, config.strategy_params)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else config.symbols

    data = MarketData(creds, feed=config.data.feed)
    lookback_days = max(args.days, lookback_days_for(config.timeframe, strategy.min_bars + 50))
    bars = data.get_bars(symbols, timeframe=config.timeframe, lookback_days=lookback_days)

    if not bars:
        print("No data returned; cannot backtest.")
        return 1

    print(f"\nStrategy: {strategy.name}  |  ${args.cash:,.0f} starting equity  |  {lookback_days}d window  |  timeframe {config.timeframe}\n")
    header = (
        f"{'SYMBOL':<8}{'BARS':>7}{'TRADES':>8}{'WIN%':>7}{'FINAL':>13}"
        f"{'RETURN':>9}{'SHARPE':>8}{'MAXDD':>8}{'EXP(R)':>9}{'KELLY':>8}"
    )
    print(header)
    print("-" * len(header))

    for symbol, df in bars.items():
        if len(df) <= strategy.min_bars:
            print(f"{symbol:<8}{len(df):>7}  (not enough bars, need > {strategy.min_bars})")
            continue

        result = run_backtest(
            strategy, df, symbol, config.risk,
            timeframe=config.timeframe, starting_equity=args.cash,
        )
        print(
            f"{result.symbol:<8}{len(df):>7}{len(result.trades):>8}"
            f"{result.expectancy.win_rate:>7.1%}{result.final_equity:>13,.2f}"
            f"{result.total_return_pct:>9.1%}{result.sharpe_ratio:>8.2f}"
            f"{result.max_drawdown_pct:>8.1%}{result.expectancy.expectancy_r:>9.2f}"
            f"{result.kelly_fraction:>8.1%}"
        )

        if result.trades:
            trade_path = save_trade_log(symbol, strategy.name, result.trades)
            equity_path = save_equity_curve(symbol, strategy.name, result.equity_curve)
            print(f"         -> {trade_path.relative_to(ROOT)}, {equity_path.relative_to(ROOT)}")

    print(
        "\nBreakeven win rate and margin of safety are per-symbol; see the printed "
        "EXP(R) (expectancy in R-multiples) for the headline number. Slippage "
        "0.02% and $0 commission are modelled; spread and partial fills are not.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
