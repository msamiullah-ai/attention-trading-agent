"""Entry point: orchestrates config, credentials, the Trader, and the dashboard.

    python main.py                          # continuous paper-trading loop with dashboard
    python main.py --dry-run                # log intended orders only, nothing sent
    python main.py --once                   # single cycle then exit (handy for cron/testing)
    python main.py --interval 30            # override config.yaml's poll interval
    python main.py --config config_crypto.yaml   # run a second, independent bot off a
                                                    # different config -- e.g. a
                                                    # crypto-only instance alongside the
                                                    # default equity one, each its own process

For account inspection, closing a single position, or an emergency flatten,
see run.py (status / close / panic) — this file is the trading loop only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot.config import PROJECT_ROOT, load_config, load_credentials  # noqa: E402
from trading_bot.dashboard import print_dashboard  # noqa: E402
from trading_bot.logger import get_logger, setup_logging  # noqa: E402
from trading_bot.trader import Trader  # noqa: E402

log = get_logger("main")


def main() -> int:
    parser = argparse.ArgumentParser(description="Alpaca auto-trader", epilog=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log intended orders, send nothing")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--interval", type=int, default=None, help="Seconds between cycles")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a config.yaml variant (default: config.yaml). Lets a second bot "
             "instance run off different symbols/strategy without touching the default.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        creds = load_credentials()
        config_path = (PROJECT_ROOT / args.config) if args.config else None
        config = load_config(config_path)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    if not creds.paper and not args.dry_run:
        confirm = input("\n*** LIVE TRADING WITH REAL MONEY. Type 'I ACCEPT' to continue: ")
        if confirm.strip() != "I ACCEPT":
            print("Aborted.")
            return 1

    trader = Trader(config, creds, dry_run=args.dry_run)

    if args.once:
        trader.run_cycle()
        print_dashboard(trader)
        return 0

    trader.run_forever(args.interval, on_cycle=lambda: print_dashboard(trader))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
