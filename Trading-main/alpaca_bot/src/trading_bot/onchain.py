"""On-chain / sentiment signal layer: a FILTER that can adjust position
sizing for a crypto strategy, not a standalone strategy.

Only the Crypto Fear & Greed Index is implemented -- it's the one source in
the original design that's genuinely free and keyless (alternative.me).
Exchange netflow and stablecoin-reserve signals (CryptoQuant/Glassnode) need
a paid tier or an API key even on their "free" plans, so they're left as an
extension point rather than faked.

Deliberately NOT wired into any Strategy's evaluate(): strategies here are
pure and deterministic (same input bars -> same signal, every time), which
is what lets backtest.py precompute indicators once and replay thousands of
bars fast. A live network call per bar would break both properties -- it'd
make backtests non-reproducible and, called inside the bar-by-bar loop,
turn a few-second backtest into one bounded by the Fear & Greed API's
latency times the bar count. If you want this to influence live sizing,
call `fetch_fear_greed()` / `position_size_multiplier()` once per cycle in
trader.py (e.g. scaling `decision.qty` in `_maybe_enter`), not per bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from .logger import get_logger

log = get_logger(__name__)

_FNG_URL = "https://api.alternative.me/fng/"


@dataclass
class FearGreedReading:
    value: int  # 0 (extreme fear) - 100 (extreme greed)
    classification: str  # e.g. "Fear", "Greed", "Extreme Fear"


def fetch_fear_greed(timeout_s: float = 5.0) -> FearGreedReading | None:
    """Latest Crypto Fear & Greed Index reading, or None if unreachable.

    Degrades gracefully by design (per the module docstring) -- callers
    should treat None as "no adjustment", not raise it further.
    """
    try:
        resp = requests.get(_FNG_URL, params={"limit": 1}, timeout=timeout_s)
        resp.raise_for_status()
        entry = resp.json()["data"][0]
        return FearGreedReading(
            value=int(entry["value"]),
            classification=entry["value_classification"],
        )
    except Exception as exc:  # noqa: BLE001 - any failure here must not break trading
        log.warning("Could not fetch Fear & Greed Index: %s", exc)
        return None


def position_size_multiplier(
    reading: FearGreedReading | None,
    extreme_fear_below: int = 20,
    extreme_greed_above: int = 80,
    fear_boost: float = 1.2,
    greed_cut: float = 0.5,
) -> float:
    """Scale a position size by crowd sentiment: extreme fear is historically
    a contrarian buy signal (boost size a bit); extreme greed means the
    market is overheated (cut size). 1.0 (no adjustment) if `reading` is
    None or sentiment is unremarkable.
    """
    if reading is None:
        return 1.0
    if reading.value <= extreme_fear_below:
        return fear_boost
    if reading.value >= extreme_greed_above:
        return greed_cut
    return 1.0
