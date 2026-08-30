"""Backtest LooseEmaRsiStrategy against real history, side-by-side with the
strict EmaRsiStrategy it's adapted from, plus the same randomization test
used on every other strategy this session (does its entry timing actually
beat random, or is it indistinguishable from noise?).

Pure addition -- imports run_backtest/MarketData/etc. from the existing,
untouched modules; doesn't register anything in signals.REGISTRY, doesn't
modify signals.py, backtest.py, or any other existing file.

    python scripts/backtest_loose_ema_rsi.py --days 180 --symbols AAPL,NVDA,MSFT
    python scripts/backtest_loose_ema_rsi.py --days 180 --trials 30
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_bot.backtest import run_backtest  # noqa: E402
from trading_bot.config import load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData, lookback_days_for  # noqa: E402
from trading_bot.logger import setup_logging  # noqa: E402
from trading_bot.signals import BUY, EmaRsiStrategy, HOLD  # noqa: E402
from trading_bot.strategies_experimental import LooseEmaRsiStrategy  # noqa: E402


class RandomEntryWrapper:
    """Minimal stand-in matching the Strategy interface's essentials, so the
    randomization test (same idea as scripts/validate_strategy.py) can run
    without importing that script's internals."""

    def __init__(self, base, entry_bars: dict[int, str]):
        self.base = base
        self.name = f"{base.name}_random"
        self.min_bars = base.min_bars
        self.allow_short = base.allow_short
        self.entry_bars = entry_bars

    def precompute(self, df):
        return {}

    def evaluate(self, df, ctx, i):
        return self.entry_bars.get(i, HOLD)

    def get_stop_loss(self, entry_price, df, side):
        return self.base.get_stop_loss(entry_price, df, side)

    def get_take_profit(self, entry_price, df, side):
        return self.base.get_take_profit(entry_price, df, side)


def randomization_check(strategy, df, symbol, risk_config, timeframe, cash, trials, rng):
    real = run_backtest(strategy, df, symbol, risk_config, timeframe=timeframe, starting_equity=cash)
    n_entries = len(real.trades)
    if n_entries == 0:
        return real, None

    eligible = range(strategy.min_bars, len(df))
    random_exps = []
    for _ in range(trials):
        chosen = rng.sample(eligible, k=min(n_entries, len(eligible)))
        wrapper = RandomEntryWrapper(strategy, {i: BUY for i in chosen})
        trial = run_backtest(wrapper, df, symbol, risk_config, timeframe=timeframe, starting_equity=cash)
        random_exps.append(trial.expectancy.expectancy_r)

    pctl = sum(1 for x in random_exps if x <= real.expectancy.expectancy_r) / len(random_exps)
    return real, pctl


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare loose vs strict ema_rsi, plus a randomization check")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("WARNING")
    config = load_config()
    creds = load_credentials()
    rng = random.Random(args.seed)

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else config.symbols
    data = MarketData(creds, feed=config.data.feed)

    strategies = {"ema_rsi (strict)": EmaRsiStrategy(), "loose_ema_rsi": LooseEmaRsiStrategy()}
    max_min_bars = max(s.min_bars for s in strategies.values())
    lookback_days = max(args.days, lookback_days_for(config.timeframe, max_min_bars + 50))
    bars = data.get_bars(symbols, timeframe=config.timeframe, lookback_days=lookback_days)

    if not bars:
        print("No data returned; cannot backtest.")
        return 1

    print(f"\n{lookback_days}d window, timeframe {config.timeframe}, ${args.cash:,.0f} starting equity\n")
    header = f"{'STRATEGY':<18}{'SYMBOL':<8}{'BARS':>7}{'TRADES':>8}{'WIN%':>7}{'EXP(R)':>9}{'PCTL':>7}"
    print(header)
    print("-" * len(header))

    for label, strategy in strategies.items():
        for symbol, df in bars.items():
            if len(df) <= strategy.min_bars:
                print(f"{label:<18}{symbol:<8}{len(df):>7}  (not enough bars, need > {strategy.min_bars})")
                continue
            # Reusing `strategy` across symbols is safe: run_backtest builds
            # its own fresh in-memory RiskManager per call, so no trade
            # history or sizing state carries over between symbols.
            result, pctl = randomization_check(
                strategy, df, symbol, config.risk, config.timeframe, args.cash, args.trials, rng
            )
            pctl_str = f"{pctl:.0%}" if pctl is not None else "n/a"
            print(
                f"{label:<18}{symbol:<8}{len(df):>7}{len(result.trades):>8}"
                f"{result.expectancy.win_rate:>7.1%}{result.expectancy.expectancy_r:>9.2f}{pctl_str:>7}"
            )

    print(
        "\nPCTL = where this strategy's expectancy falls within a random-entry "
        "distribution using the same exits and the same number of entries. "
        "~50% means the entry timing isn't adding anything over chance -- "
        "same interpretation as scripts/validate_strategy.py.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
