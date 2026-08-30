"""Smoke test: verify credentials, trading API and market-data API all work.

Run this first, before anything else:
    python scripts/check_connection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_bot.broker import Broker  # noqa: E402
from trading_bot.config import load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData  # noqa: E402
from trading_bot.logger import setup_logging  # noqa: E402


def main() -> int:
    setup_logging("INFO")
    failures = 0

    print("\n[1/4] Loading credentials...")
    try:
        creds = load_credentials()
        print(f"      OK - mode: {creds.mode}")
        if not creds.paper:
            print("      WARNING: configured for LIVE trading with real money.")
    except RuntimeError as exc:
        print(f"      FAIL - {exc}")
        return 1

    print("[2/4] Loading config.yaml...")
    try:
        config = load_config()
        print(f"      OK - strategy={config.strategy} symbols={','.join(config.symbols)}")
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL - {exc}")
        return 1

    print("[3/4] Connecting to Trading API...")
    try:
        broker = Broker(creds)
        account = broker.get_account()
        print(f"      OK - equity ${account.equity:,.2f}, market open: {broker.is_market_open()}")
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL - {exc}")
        print("      Check that your keys match the mode (paper keys != live keys).")
        failures += 1

    print("[4/4] Fetching market data...")
    try:
        data = MarketData(creds, feed=config.data.feed)
        symbol = config.symbols[0]
        bars = data.get_bars([symbol], timeframe="1Day", lookback_days=30)
        if symbol in bars and not bars[symbol].empty:
            df = bars[symbol]
            print(f"      OK - {len(df)} daily bars for {symbol}, last close ${df['close'].iloc[-1]:,.2f}")
        else:
            print(f"      FAIL - no bars returned for {symbol}")
            failures += 1
    except Exception as exc:  # noqa: BLE001
        print(f"      FAIL - {exc}")
        failures += 1

    print("\nAll checks passed.\n" if not failures else f"\n{failures} check(s) failed.\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
