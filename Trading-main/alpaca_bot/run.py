"""Command-line entrypoint for account/position utilities.

Examples:
    python run.py status                 # account + positions, places no orders
    python run.py trade --dry-run        # one cycle, log intended orders only
    python run.py trade                  # one cycle, place real orders
    python run.py trade --loop           # run continuously (see main.py for the dedicated loop)
    python run.py close AAPL             # liquidate one position
    python run.py panic                  # cancel orders + close everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot.broker import Broker  # noqa: E402
from trading_bot.config import load_config, load_credentials  # noqa: E402
from trading_bot.dashboard import print_dashboard  # noqa: E402
from trading_bot.logger import get_logger, setup_logging  # noqa: E402
from trading_bot.trader import Trader  # noqa: E402

log = get_logger("run")


def cmd_status(args) -> int:
    creds = load_credentials()
    broker = Broker(creds)
    account = broker.get_account()

    print(f"\n=== ACCOUNT ({creds.mode}) ===")
    print(f"  Equity:        ${account.equity:,.2f}")
    print(f"  Cash:          ${account.cash:,.2f}")
    print(f"  Buying power:  ${account.buying_power:,.2f}")
    print(f"  Day P/L:       {account.daily_pnl_pct:+.2%}")
    print(f"  Market open:   {broker.is_market_open()}")

    positions = broker.get_positions()
    print(f"\n=== POSITIONS ({len(positions)}) ===")
    if not positions:
        print("  (none)")
    for p in positions.values():
        print(
            f"  {p.symbol:<6} qty={p.qty:<8g} @ ${p.avg_entry_price:>9,.2f}  "
            f"value=${p.market_value:>11,.2f}  P/L=${p.unrealized_pl:>+10,.2f} "
            f"({p.unrealized_plpc:+.2%})"
        )

    orders = broker.get_open_orders()
    print(f"\n=== OPEN ORDERS ({len(orders)}) ===")
    if not orders:
        print("  (none)")
    for o in orders:
        print(f"  {o.symbol:<6} {o.side.value:<4} {o.qty} {o.order_type.value} [{o.status.value}]")
    print()
    return 0


def cmd_trade(args) -> int:
    creds = load_credentials()
    config = load_config()

    if not creds.paper and not args.dry_run:
        confirm = input(
            "\n*** LIVE TRADING WITH REAL MONEY. Type 'I ACCEPT' to continue: "
        )
        if confirm.strip() != "I ACCEPT":
            print("Aborted.")
            return 1

    trader = Trader(config, creds, dry_run=args.dry_run)
    if args.loop:
        trader.run_forever(args.interval, on_cycle=lambda: print_dashboard(trader))
    else:
        trader.run_cycle()
        print_dashboard(trader)
    return 0


def cmd_buy(args) -> int:
    broker = Broker(load_credentials())
    order = broker.buy(args.symbol, args.qty, order_type="market", time_in_force="day")
    print(f"Buy order submitted: {args.qty} {args.symbol.upper()} (id={order.id})")
    return 0


def cmd_sell(args) -> int:
    broker = Broker(load_credentials())
    order = broker.sell(args.symbol, args.qty, order_type="market", time_in_force="day")
    print(f"Sell order submitted: {args.qty} {args.symbol.upper()} (id={order.id})")
    return 0


def cmd_close(args) -> int:
    broker = Broker(load_credentials())
    broker.close_position(args.symbol)
    print(f"Close order submitted for {args.symbol.upper()}.")
    return 0


def cmd_panic(args) -> int:
    creds = load_credentials()
    confirm = input(
        f"Cancel ALL orders and liquidate ALL positions in the {creds.mode} account? [y/N]: "
    )
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return 1
    broker = Broker(creds)
    broker.cancel_all_orders()
    broker.close_all_positions(cancel_orders=True)
    print("All orders cancelled and positions closed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alpaca trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show account, positions and open orders").set_defaults(
        func=cmd_status
    )

    p_trade = sub.add_parser("trade", help="Run the strategy")
    p_trade.add_argument("--dry-run", action="store_true", help="Log orders without sending them")
    p_trade.add_argument("--loop", action="store_true", help="Run continuously")
    p_trade.add_argument("--interval", type=int, help="Seconds between cycles when looping")
    p_trade.set_defaults(func=cmd_trade)

    p_buy = sub.add_parser("buy", help="Submit a manual market buy order")
    p_buy.add_argument("symbol")
    p_buy.add_argument("qty", type=float, nargs="?", default=1.0)
    p_buy.set_defaults(func=cmd_buy)

    p_sell = sub.add_parser("sell", help="Submit a manual market sell order")
    p_sell.add_argument("symbol")
    p_sell.add_argument("qty", type=float, nargs="?", default=1.0)
    p_sell.set_defaults(func=cmd_sell)

    p_close = sub.add_parser("close", help="Liquidate one position")
    p_close.add_argument("symbol")
    p_close.set_defaults(func=cmd_close)

    sub.add_parser("panic", help="Cancel all orders and close all positions").set_defaults(
        func=cmd_panic
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        return args.func(args)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
