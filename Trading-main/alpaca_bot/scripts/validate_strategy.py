"""Statistical validation: is a strategy's entry timing actually better/worse
than random, or is the backtest result just noise?

For each (strategy, symbol): runs the real backtest, then runs N trials where
entries fire at random bars (same count, same long/short mix, same
stop-loss/take-profit logic as the real strategy) instead of the strategy's
actual signal timing. If the real result sits in the middle of the random
distribution, the entry logic isn't adding anything over noise.

    python scripts/validate_strategy.py --days 365 --trials 20
    python scripts/validate_strategy.py --days 365 --symbols NVDA --strategy bollinger_squeeze
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_bot.backtest import resolve_starting_equity, run_backtest  # noqa: E402
from trading_bot.config import load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData, lookback_days_for  # noqa: E402
from trading_bot.logger import setup_logging  # noqa: E402
from trading_bot.signals import BUY, HOLD, SELL, Strategy, build_strategy  # noqa: E402


class RandomEntryWrapper(Strategy):
    """Same stop/target logic as `base`, but entries fire at fixed bar
    indices instead of the base strategy's actual signal timing."""

    def __init__(self, base: Strategy, entry_bars: dict[int, str]):
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


def make_random_wrapper(base: Strategy, n_long: int, n_short: int, n_bars: int, rng: random.Random) -> RandomEntryWrapper:
    eligible = range(base.min_bars, n_bars)
    n_total = min(n_long + n_short, len(eligible))
    chosen = rng.sample(eligible, k=n_total)
    sides = [BUY] * n_long + [SELL] * n_short
    rng.shuffle(sides)
    entry_bars = dict(zip(chosen, sides[:n_total]))
    return RandomEntryWrapper(base, entry_bars)


def percentile_rank(value: float, distribution: list[float]) -> float:
    if not distribution:
        return float("nan")
    return sum(1 for x in distribution if x <= value) / len(distribution)


def verdict_for(percentile: float, n_random: int) -> str:
    if n_random == 0:
        return "no random trials completed"
    if percentile >= 0.95:
        return "entry timing beats random (top 5%) -- possible real edge"
    if percentile <= 0.05:
        return "entry timing WORSE than random (bottom 5%) -- signal may be anti-correlated"
    return "indistinguishable from random entry timing -- no demonstrated edge"


def main() -> int:
    parser = argparse.ArgumentParser(description="Randomization test for a strategy's entry timing")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--cash", type=float, default=None,
                        help="Starting equity (default: your live account equity)")
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("WARNING")
    config = load_config()
    creds = load_credentials()
    rng = random.Random(args.seed)

    strategy_name = args.strategy or config.strategy
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else config.symbols

    data = MarketData(creds, feed=config.data.feed)
    base_strategy = build_strategy(strategy_name, config.strategy_params)
    lookback_days = max(args.days, lookback_days_for(config.timeframe, base_strategy.min_bars + 50))
    bars = data.get_bars(symbols, timeframe=config.timeframe, lookback_days=lookback_days)

    if not bars:
        print("No data returned; cannot validate.")
        return 1

    print(f"\nValidating: {strategy_name}  |  {lookback_days}d window  |  {args.trials} random trials per symbol\n")
    header = f"{'SYMBOL':<8}{'BARS':>7}{'REAL_N':>8}{'REAL_EXP(R)':>13}{'RAND_MEAN':>11}{'RAND_STD':>10}{'PCTL':>7}"
    print(header)
    print("-" * len(header))

    for symbol, df in bars.items():
        strategy = build_strategy(strategy_name, config.strategy_params)
        if len(df) <= strategy.min_bars:
            print(f"{symbol:<8}{len(df):>7}  (not enough bars, need > {strategy.min_bars})")
            continue

        args.cash = resolve_starting_equity(args.cash)
    real_result = run_backtest(strategy, df, symbol, config.risk, timeframe=config.timeframe, starting_equity=args.cash)
        n_long = sum(1 for t in real_result.trades if t.side == "long")
        n_short = sum(1 for t in real_result.trades if t.side == "short")

        if n_long + n_short == 0:
            print(f"{symbol:<8}{len(df):>7}{'0':>8}  (no trades fired -- nothing to validate)")
            continue

        random_expectancies = []
        for _ in range(args.trials):
            wrapper = make_random_wrapper(strategy, n_long, n_short, len(df), rng)
            trial_result = run_backtest(wrapper, df, symbol, config.risk, timeframe=config.timeframe, starting_equity=args.cash)
            random_expectancies.append(trial_result.expectancy.expectancy_r)

        pctl = percentile_rank(real_result.expectancy.expectancy_r, random_expectancies)
        rand_mean = statistics.mean(random_expectancies) if random_expectancies else float("nan")
        rand_std = statistics.stdev(random_expectancies) if len(random_expectancies) > 1 else 0.0

        print(
            f"{symbol:<8}{len(df):>7}{n_long + n_short:>8}"
            f"{real_result.expectancy.expectancy_r:>13.3f}{rand_mean:>11.3f}{rand_std:>10.3f}{pctl:>7.0%}"
        )
        print(f"         -> {verdict_for(pctl, len(random_expectancies))}")

    print(
        "\nPCTL = where the real strategy's expectancy falls within the random-entry "
        "distribution (same exits, random timing). ~50% means the entry logic isn't "
        "adding anything over chance.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
