"""Deploy a fixed dollar budget across whatever's currently signaling BUY,
using a 0/1 knapsack to pick the best-value combination that fits.

One-time manual command -- this does NOT change how the live bot (main.py)
sizes trades going forward; that still uses per-trade Kelly sizing via
RiskManager, independent of this script.

How it works:
  1. For every configured symbol, run the strategy/strategies applicable to
     its asset class (equity strategies for stocks, crypto_momentum for
     crypto) and keep whichever ones say BUY right now.
  2. Size each candidate the same way the live bot would (RiskManager's
     existing Kelly/fallback sizing) -- that's the candidate's "weight"
     (dollar cost). If a symbol gets a BUY from more than one strategy, only
     the one with the better historical expectancy is kept, since you can
     only hold one position per symbol.
  3. Each candidate's "value" is its strategy's own historical expectancy in
     R-multiples (RiskManager.expectancy(strategy=...).expectancy_r) --
     0.0 if that strategy has no trade history yet, which is the honest
     current state for everything in this repo (see the validation results
     from earlier: none of the four strategies have a proven edge yet, so
     right now this mostly picks "as many affordable candidates as fit",
     not "the highest-conviction ones" -- it'll differentiate more once
     real trade history accumulates).
  4. Run the knapsack over these candidates with your budget as the
     capacity, and submit orders for whatever it selects. Equity buys get
     real bracket-order stop/target protection, same as the live bot;
     crypto buys don't (Alpaca rejects bracket orders for crypto outright)
     and nothing here watches them afterward the way trader.py's live loop
     does -- a crypto fill from this script is unprotected until you close
     it yourself.

Positions opened this way are NOT tracked by the live bot's in-memory trade
log / expectancy system (that only tracks trades it opened itself) -- if the
live bot's strategy later emits an exit signal for one of these symbols it
may still close it (trader.py acts on any real position, not just ones it
opened), but won't record a P/L for it. Track/close these manually via
`run.py status` / `run.py close SYMBOL` if you want clean bookkeeping.

    python scripts/deploy_capital.py --budget 500
    python scripts/deploy_capital.py --budget 2000 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_bot.broker import Broker  # noqa: E402
from trading_bot.config import load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData, is_crypto_symbol, lookback_days_for  # noqa: E402
from trading_bot.knapsack import KnapsackItem, knapsack_select  # noqa: E402
from trading_bot.logger import setup_logging  # noqa: E402
from trading_bot.risk import RiskManager  # noqa: E402
from trading_bot.signals import BUY, REGISTRY, Strategy, build_strategy  # noqa: E402

EQUITY_STRATEGIES = ["ema_rsi", "vwap_mean_reversion", "bollinger_squeeze"]
CRYPTO_STRATEGIES = ["crypto_momentum"]


class Candidate:
    def __init__(
        self, symbol: str, strategy: Strategy, qty: float, price: float,
        expectancy_r: float, stop: float, target: float,
    ):
        self.symbol = symbol
        self.strategy = strategy
        self.qty = qty
        self.price = price
        self.cost = qty * price
        self.expectancy_r = expectancy_r
        self.stop = stop
        self.target = target

    @property
    def weight_dollars(self) -> int:
        return max(1, round(self.cost))  # at least $1 so a real cost is never "free" in the DP


def build_candidates(config, data: MarketData, risk: RiskManager, account) -> list[Candidate]:
    equities = [s for s in config.symbols if not is_crypto_symbol(s)]
    cryptos = [s for s in config.symbols if is_crypto_symbol(s)]

    max_min_bars = max(REGISTRY[name]().min_bars for name in REGISTRY)
    lookback_days = lookback_days_for(config.timeframe, max_min_bars + 50)
    bars = data.get_bars(config.symbols, timeframe=config.timeframe, lookback_days=lookback_days)

    candidates_by_symbol: dict[str, Candidate] = {}
    plan = [(s, EQUITY_STRATEGIES) for s in equities] + [(s, CRYPTO_STRATEGIES) for s in cryptos]

    for symbol, strategy_names in plan:
        df = bars.get(symbol)
        if df is None:
            continue
        for strategy_name in strategy_names:
            strategy = build_strategy(strategy_name)
            if len(df) < strategy.min_bars:
                continue
            if strategy.generate_signal(df) != BUY:
                continue

            price = data.get_last_price(symbol) or float(df["close"].iloc[-1])
            qty = risk.size_position(account, price, strategy_name, symbol)
            min_qty = 1e-6 if is_crypto_symbol(symbol) else 1.0
            if qty < min_qty:
                continue

            expectancy_r = risk.expectancy(strategy=strategy_name).expectancy_r
            stop = strategy.get_stop_loss(price, df, "long")
            target = strategy.get_take_profit(price, df, "long")
            candidate = Candidate(symbol, strategy, qty, price, expectancy_r, stop, target)

            existing = candidates_by_symbol.get(symbol)
            if existing is None or candidate.expectancy_r > existing.expectancy_r:
                candidates_by_symbol[symbol] = candidate

    return list(candidates_by_symbol.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Knapsack-allocate a budget across current BUY signals")
    parser.add_argument("--budget", type=float, required=True, help="Total dollars to deploy")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan, submit nothing")
    args = parser.parse_args()

    setup_logging("WARNING")
    config = load_config()
    creds = load_credentials()

    broker = Broker(creds)
    data = MarketData(creds, feed=config.data.feed)
    risk = RiskManager(config.risk)
    account = broker.get_account()

    print(f"\nScanning {len(config.symbols)} symbols for BUY signals across {len(REGISTRY)} strategies...")
    candidates = build_candidates(config, data, risk, account)

    if not candidates:
        print("No BUY signals right now across any applicable strategy. Nothing to deploy.\n")
        return 0

    open_positions = broker.get_positions()
    candidates = [c for c in candidates if c.symbol not in open_positions]
    if not candidates:
        print("All current BUY signals are for symbols you already hold. Nothing to deploy.\n")
        return 0

    items = [
        KnapsackItem(id=c.symbol, weight=c.weight_dollars, value=c.expectancy_r)
        for c in candidates
    ]
    budget_dollars = max(0, round(args.budget))
    selected_ids = {i.id for i in knapsack_select(items, budget_dollars)}
    selected = [c for c in candidates if c.symbol in selected_ids]

    print(f"\n{len(candidates)} candidate(s) found, budget ${args.budget:,.2f}:\n")
    header = f"{'SYMBOL':<10}{'STRATEGY':<24}{'QTY':>14}{'PRICE':>14}{'COST':>12}{'EXP(R)':>9}{'PICKED':>8}"
    print(header)
    print("-" * len(header))
    for c in sorted(candidates, key=lambda c: -c.expectancy_r):
        picked = "YES" if c in selected else ""
        print(
            f"{c.symbol:<10}{c.strategy.name:<24}{c.qty:>14.6f}{c.price:>14,.2f}"
            f"{c.cost:>12,.2f}{c.expectancy_r:>9.2f}{picked:>8}"
        )

    total_cost = sum(c.cost for c in selected)
    print(f"\nSelected {len(selected)} position(s), total cost ${total_cost:,.2f} of ${args.budget:,.2f} budget.")

    if not selected:
        print("Nothing fit the budget (cheapest candidate may exceed it).\n")
        return 0

    if args.dry_run:
        print("[DRY RUN] No orders submitted.\n")
        return 0

    if not creds.paper:
        confirm = input("\n*** LIVE TRADING WITH REAL MONEY. Type 'I ACCEPT' to continue: ")
        if confirm.strip() != "I ACCEPT":
            print("Aborted.")
            return 1

    for c in selected:
        crypto = is_crypto_symbol(c.symbol)
        # Equities get real bracket-order protection (stop/target live
        # server-side at Alpaca). Crypto can't -- broker.py already knows to
        # drop the bracket legs for it (Alpaca rejects them outright) and
        # submits a plain order instead; nothing here monitors that stop
        # afterward the way trader.py's live loop does, so treat a crypto
        # fill from this script as unprotected until you close it yourself.
        broker.buy(
            c.symbol, c.qty,
            order_type=config.execution.order_type,
            time_in_force="gtc" if crypto else config.execution.time_in_force,
            take_profit_price=c.target,
            stop_loss_price=c.stop,
        )
        protection = "unprotected (crypto, no bracket support)" if crypto else f"stop={c.stop:.2f} target={c.target:.2f}"
        print(f"  BUY {c.symbol} x{c.qty:.6f} submitted ({protection})")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
