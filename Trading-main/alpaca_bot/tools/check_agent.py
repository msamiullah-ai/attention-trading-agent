"""Exercise the agent layer against the real world and print what it decided.

WHY THIS EXISTS

The test suite proves the plumbing: that a veto stops an order, that a shrink
reduces quantity, that a confirm cannot enlarge one. Every one of those tests
feeds the parser a canned string. None of them has ever run a language model,
touched a live chain, or read a real headline -- so none of them can tell you
whether the model produces SENSIBLE vetoes, only that a veto would be obeyed.

This script closes that gap. It uses real credentials, real bars, real news and
a real model, and prints the reasoning rather than asserting on it, because the
question it answers is "is this judgement any good" and that is a question for
a human to look at, not for an assert.

It places no orders. Every trade it considers is hypothetical.

    python tools/check_agent.py                # everything that is configured
    python tools/check_agent.py --symbols AAPL,NVDA
    python tools/check_agent.py --advisor-only # skip bars, just probe the model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from trading_bot.agent import build_advisor, build_prompt, review  # noqa: E402
from trading_bot.attention import attend  # noqa: E402
from trading_bot.config import load_agent_settings, load_config, load_credentials  # noqa: E402
from trading_bot.data import MarketData, lookback_days_for  # noqa: E402
from trading_bot.regime import detect_regime  # noqa: E402
from trading_bot.trader import Trader  # noqa: E402

BAR = "=" * 72


def head(text: str) -> None:
    print(f"\n{BAR}\n{text}\n{BAR}")


def check_keys() -> dict[str, bool]:
    head("1. CREDENTIALS")
    settings = load_agent_settings()
    have = {}
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        value = os.getenv(name, "").strip()
        have[name] = bool(value)
        # Length only. Never print a key, not even a prefix -- a truncated key
        # in a terminal is still a key in scrollback and in any shared screen.
        print(f"  {name:22} {(f'set ({len(value)} chars)' if value else 'MISSING'):20} "
              f"needed for market data")
    print(f"  {'ALPACA_PAPER':22} {os.getenv('ALPACA_PAPER', 'true')}")

    # Everything else comes from the one settings object, already redacted.
    print("\n  agent layer (from load_agent_settings):")
    for key, value in settings.redacted().items():
        print(f"    {key:20} {value}")
    return have


def check_fusion(symbols: list[str]) -> None:
    """Run the real strategies on real bars and show who voted for what."""
    head("2. SIGNAL FUSION (real bars, real indicators)")
    try:
        config, creds = load_config(), load_credentials()
    except Exception as exc:
        print(f"  cannot load config/credentials: {exc}")
        return

    data = MarketData(creds, feed=config.data.feed)
    trader = Trader(config, creds, dry_run=True, fusion=True)
    lookback = lookback_days_for(config.timeframe, config.lookback_bars)

    for symbol in symbols:
        print(f"\n  --- {symbol} ---")
        try:
            bars = data.get_bars([symbol], timeframe=config.timeframe,
                                 lookback_days=lookback)
            df = bars.get(symbol)
        except Exception as exc:
            print(f"  bar fetch failed: {exc}")
            continue
        if df is None or df.empty:
            print("  no bars returned")
            continue

        regime = detect_regime(df)
        votes = trader._collect_votes(symbol, df)
        print(f"  bars={len(df)}  regime={regime}")
        if not votes:
            print("  every strategy abstained (not enough history)")
            continue
        for name, vote in votes.items():
            print(f"    {name:22} -> {vote}")

        verdict = attend(votes, regime=regime, temperature=trader.fusion_temperature)
        print(f"  FUSED: {verdict.explain()}")

        # The comparison that shows fusion is doing something: what the single
        # configured strategy would have said on its own.
        solo = trader.strategy.generate_signal(df) if len(df) >= trader.strategy.min_bars else "n/a"
        flag = "  <-- DIFFERS" if solo != verdict.action else ""
        print(f"  solo ({trader.strategy.name}): {solo}{flag}")


# Hypothetical trades chosen to span the cases the prompt cares about. None of
# these is placed; they exist to see whether the model distinguishes a real,
# dated, specific reason from vague unease -- which is the entire ask.
SCENARIOS = [
    ("AAPL", "buy", 50, 180.0,
     "EVENTS: no earnings inside the holding window\nNEWS: none returned for this ticker in the window",
     "expect CONFIRM - nothing to object to, and silence is not a warning"),
    ("NVDA", "buy", 20, 900.0,
     "EVENTS: earnings for NVDA in 2d (2026-09-05); premium collapses but the stock gaps\n"
     "NEWS (1 items, newest first):\n  - [2026-09-01] NVDA to report Q3 results Thursday after the close",
     "expect VETO or SHRINK - a specific, dated, binary event"),
    ("XYZ", "buy", 100, 12.0,
     "EVENTS: no earnings inside the holding window\n"
     "NEWS (2 items, newest first):\n"
     "  - [2026-09-02] XYZ halted pending news; company confirms strategic review\n"
     "  - [2026-09-02] Analysts flag going-concern language in latest filing",
     "expect VETO - halt plus going-concern doubt"),
    ("SPY", "buy", 10, 550.0,
     "EVENTS: no earnings inside the holding window\n"
     "NEWS (1 items, newest first):\n"
     "  - [2026-09-02] Investors nervous as markets look expensive amid uncertainty",
     "expect CONFIRM - vague macro worry is explicitly not grounds to veto"),
    ("TSLA", "buy", 30, 250.0,
     "EVENTS: event calendar unavailable; treated as unknown, not as clear\n"
     "NEWS (1 items, newest first):\n  - [2026-09-02] TSLA surges 14% on delivery beat",
     "expect SHRINK or CONFIRM - the move already happened, signal may be stale"),
]


def probe_endpoint() -> None:
    """Confirm the endpoint answers and the model name is one it recognises.

    Worth doing before the scenarios, because "key is wrong", "URL is wrong" and
    "model name is a typo" all surface downstream as the same thing: an advisor
    that quietly declines to run. Separating them here saves guessing later.
    """
    import json as _json
    import urllib.error
    import urllib.request

    settings = load_agent_settings()
    base, key, want = settings.base_url, settings.api_key, settings.model
    if not base:
        return

    print(f"\n  endpoint: {base}")
    print(f"  model:    {want or '(LLM_MODEL is empty)'}")
    # Same User-Agent requirement as the advisor itself -- without it a
    # Cloudflare-fronted provider 403s the probe and the diagnostic lies.
    from trading_bot.agent import USER_AGENT
    headers = {"User-Agent": USER_AGENT}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        request = urllib.request.Request(f"{base}/models", headers=headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            data = _json.loads(response.read().decode())
        names = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
        print(f"  reachable, {len(names)} models listed")
        if want and names:
            if want in names:
                print("  model name matches an available model")
            else:
                near = [n for n in names if want.split("/")[-1].lower() in n.lower()][:5]
                print(f"  WARNING: {want!r} is not in the served list")
                if near:
                    print("  close matches: " + ", ".join(near))
    except urllib.error.HTTPError as exc:
        hint = {401: "key rejected", 403: "key lacks access", 404: "wrong base URL"}
        print(f"  HTTP {exc.code} - {hint.get(exc.code, 'see response')}")
    except Exception as exc:
        print(f"  unreachable: {exc}")


def check_advisor() -> None:
    head("3. LLM REVIEWER (real model, hypothetical trades, no orders placed)")
    advisor = build_advisor(load_agent_settings())
    if advisor is None:
        print("  No advisor built. TRADING_BOT_ADVISOR is 'none', or the model/key")
        print("  it needs is unset. The bot runs rules-only in this state, which is")
        print("  a valid mode -- just not the one you are trying to test.")
        return

    probe_endpoint()

    agree = 0
    unparsed = 0
    for symbol, side, qty, price, context, expectation in SCENARIOS:
        advice = review(advisor, symbol, side, qty, price, context)
        matched = advice.action.upper() in expectation.upper()
        agree += matched
        unparsed += bool(advice.error)
        print(f"\n  --- {symbol} {side} {qty} @ {price} ---")
        print(f"  {expectation}")
        print(f"  GOT: {advice.action.upper()}"
              f"{f' x{advice.size_factor:.2f}' if advice.action == 'shrink' else ''}"
              f"  (confidence {advice.confidence:.2f})  {'OK' if matched else 'DIFFERS'}")
        print(f"  reason: {advice.reason or '(none given)'}")
        if advice.error:
            print(f"  note: {advice.error}")
        print(f"  qty {qty} -> {advice.apply_to(qty)}")

    print(f"\n  Matched expectation on {agree}/{len(SCENARIOS)}.")
    print("  These expectations are judgement calls, not a spec. A disagreement")
    print("  is worth reading, not automatically a bug -- but a model that vetoes")
    print("  the vague-macro case or confirms the halted one is miscalibrated.")

    if unparsed:
        print(f"\n  *** {unparsed}/{len(SCENARIOS)} replies could not be parsed. ***")
        print("  This is THE failure mode to watch for with open-weight models.")
        print("  Unparseable replies are treated as confirms by design -- a garbled")
        print("  answer is not an objection -- so a model that never returns clean")
        print("  JSON does not break the bot, it just silently reviews nothing.")
        print("  Fix it by choosing a stronger instruction-following model rather")
        print("  than by loosening the parser.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA")
    ap.add_argument("--advisor-only", action="store_true")
    ap.add_argument("--fusion-only", action="store_true")
    args = ap.parse_args()

    have = check_keys()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not args.advisor_only:
        if have["ALPACA_API_KEY"] and have["ALPACA_SECRET_KEY"]:
            check_fusion(symbols)
        else:
            head("2. SIGNAL FUSION")
            print("  skipped: Alpaca keys not set")

    if not args.fusion_only:
        check_advisor()

    head("DONE - no orders were placed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
