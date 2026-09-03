"""Entry point for the options overlay. The equity loop is main.py, untouched.

    python main_options.py                  # continuous loop over the MCP server
    python main_options.py --once           # a single cycle, then exit
    python main_options.py --dry-run        # decide everything, send nothing
    python main_options.py --interval 300   # seconds between cycles
    python main_options.py --check          # prove the MCP wiring, trade nothing

Symbols, strategy and timeframe come from config.yaml, so this takes its
DIRECTION from the same signal the equity bot uses and only the expression
differs. Risk comes from options.yaml, because a cash-secured put's exposure is
strike x 100 of posted collateral and the equity risk model has no way to say
that.

Run `--check` first, every time, on a new account. It resolves the MCP tools,
prints the approved options level, and places nothing. The level is the thing
that decides whether any of this is reachable at all:

    Level 1   sell cash-secured puts, sell covered calls
    Level 2   the above, plus buying calls and puts
    Level 3   spreads

At Level 1 a bearish signal on a name you do not own has no expression, and the
overlay will say so rather than failing silently.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot.config import PROJECT_ROOT, load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData  # noqa: E402
from trading_bot.logger import get_logger, setup_logging  # noqa: E402
from trading_bot.mcp_broker import MCPBroker, MCPError  # noqa: E402
from trading_bot import allocation
from trading_bot.options_trader import (  # noqa: E402
    OptionsTrader,
    load_options_config,
)

log = get_logger("main_options")


def preflight(broker: MCPBroker) -> int:
    """Prove the wiring before trusting it with an order.

    Everything here is a read. It answers the three questions that otherwise
    only surface as a confusing rejection mid-session: did the server start,
    which tools does this version actually expose, and what is this account
    allowed to trade?
    """
    print(f"\n{len(broker.available)} tools exposed, {len(broker.resolved)} resolved\n")
    for logical in sorted(broker.resolved):
        print(f"  [ok  ] {logical:<18} -> {broker.resolved[logical]}")
    missing = sorted(set(("place_option", "option_chain")) - set(broker.resolved))
    for logical in missing:
        print(f"  [MISS] {logical:<18} -> options trading is NOT possible")

    account = broker.get_account()
    level = broker.options_level()
    print(f"\naccount     equity {account.equity:,.2f}   cash {account.cash:,.2f}")
    print(f"            blocked: {account.blocked}")
    print(f"options     approved level: {level if level is not None else 'UNKNOWN'}")

    if level is None:
        print("\n  !! The account will not report an options level. The overlay")
        print("  !! refuses to trade rather than guess a level it may not have.")
        return 1
    if level < 1:
        print("\n  !! Level 0 -- this account cannot trade options at all.")
        return 1
    print("            level 1: cash-secured puts and covered calls"
          + ("\n            level 2: also buying calls and puts" if level >= 2 else ""))

    positions = broker.get_positions()
    print(f"\npositions   {len(positions)} open")
    for sym, p in sorted(positions.items())[:10]:
        print(f"  {sym:<24} qty {p.qty:>8g}  mv {p.market_value:>12,.2f}")
    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpaca options overlay",
                                     epilog=__doc__)
    parser.add_argument("--once", action="store_true", help="One cycle, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Decide everything, send nothing")
    parser.add_argument("--check", action="store_true",
                        help="Prove the MCP wiring and account level; trade nothing")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between cycles (default 300)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--config", type=str, default=None,
                        help="config.yaml variant for symbols/strategy")
    parser.add_argument("--options-config", type=str, default=None,
                        help="options.yaml variant for the overlay's risk limits")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        creds = load_credentials()
        cfg = load_config((PROJECT_ROOT / args.config) if args.config else None)
        opt = load_options_config(
            (PROJECT_ROOT / args.options_config) if args.options_config else None)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    # The same confirmation main.py demands, for the same reason. An options
    # overlay on a live account writes obligations, not just positions.
    if not creds.paper and not args.dry_run:
        confirm = input("\n*** LIVE TRADING WITH REAL MONEY. Type 'I ACCEPT' to continue: ")
        if confirm.strip() != "I ACCEPT":
            print("Aborted.")
            return 1

    try:
        broker = MCPBroker(creds)
    except MCPError as exc:
        log.error("could not reach the MCP server: %s", exc)
        log.error("check that `alpaca-mcp-server` is installed and on PATH")
        return 1

    if args.check:
        return preflight(broker)

    # Mirror of main.py: take only this side's share when the allocator is on.
    # The options side asks for the OPTIONS budget, so the two processes cannot
    # both spend the same dollars.
    collateral_budget = None
    if cfg.execution.allocator == "dynamic":
        account = broker.get_account()
        alloc = allocation.decide(account.equity,
                                  options_level=broker.options_level())
        collateral_budget = alloc.options_budget
        log.info("ALLOCATION %s", alloc.summary())

    trader = OptionsTrader(cfg, creds, broker=broker,
                           collateral_budget=collateral_budget,
                           data=MarketData(creds, feed=cfg.data.feed),
                           options=opt)
    if args.dry_run:
        # Dry run stops one step BEFORE submission, so it proves the decision
        # and nothing about the order path. Every defect this project has hit
        # in execution lived past this point, which is why --check exists too.
        trader.broker.submit_option_order = _refuse_to_send
        log.info("dry run: decisions will be logged, no order will be sent")

    log.info("options overlay starting: %d symbols, strategy %s, interval %ds",
             len(cfg.symbols), cfg.strategy, args.interval)

    try:
        while True:
            try:
                report = trader.run_cycle()
                log.info("cycle: %s | equity %.2f | %d option position(s) | "
                         "collateral posted %.2f",
                         report.summary(), report.equity, report.open_options,
                         report.collateral_posted)
                for row in report.skipped:
                    log.info("  skipped %s: %s", row.get("symbol", ""), row["reason"])
            except Exception as exc:  # noqa: BLE001
                # A bad cycle must not end the run. Logged and retried, the
                # same way the equity loop treats one.
                log.exception("cycle raised: %s", exc)

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped by user")
        return 0


def _refuse_to_send(**kwargs):
    log.info("DRY RUN would send: %s", kwargs)
    return {"status": "dry_run", "client_order_id": None}


if __name__ == "__main__":
    raise SystemExit(main())
