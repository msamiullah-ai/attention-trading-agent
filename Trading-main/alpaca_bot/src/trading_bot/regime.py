"""Market regime detection: classify recent price action as BULL, BEAR, or
SIDEWAYS, so a strategy only trades when the regime actually suits its
design. Nothing currently stops crypto_momentum (a breakout strategy) from
trading into a dead-flat, chopping market where breakouts are mostly noise --
this is what catches that.

Logic (ADX for trend strength, EMA position + slope for direction):
  ADX < sideways_adx                                        -> SIDEWAYS
  ADX > trending_adx, price > EMA and EMA rising              -> BULL
  ADX > trending_adx, price < EMA and EMA falling              -> BEAR
  otherwise (including the ADX 20-25 gray zone)                -> SIDEWAYS
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind

BULL = "BULL"
BEAR = "BEAR"
SIDEWAYS = "SIDEWAYS"

# Each strategy declares which regimes it's designed for.
#
# ema_rsi requires price above its own EMA20 to enter at all, so it's
# already implicitly an uptrend-only strategy -- BULL only.
# vwap_mean_reversion fades price back toward VWAP; that setup is
# characteristic of a range, not a trend -- SIDEWAYS only.
# bollinger_squeeze and crypto_momentum are both breakout/momentum designs
# that need an actual trend to follow through on, not chop -- BULL or BEAR.
STRATEGY_REGIMES: dict[str, set[str]] = {
    "ema_rsi": {BULL},
    "vwap_mean_reversion": {SIDEWAYS},
    "bollinger_squeeze": {BULL, BEAR},
    "crypto_momentum": {BULL, BEAR},
}


def detect_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    ema_period: int = 50,
    sideways_adx: float = 20.0,
    trending_adx: float = 25.0,
    slope_lookback: int = 5,
) -> str:
    min_bars = max(adx_period, ema_period, slope_lookback) + 2
    if len(df) < min_bars:
        return SIDEWAYS  # not enough data to claim a trend exists

    adx_val = ind.adx(df, adx_period).iloc[-1]
    if pd.isna(adx_val):
        return SIDEWAYS

    ema_line = ind.ema(df, ema_period)
    price = df["close"].iloc[-1]
    ema_now = ema_line.iloc[-1]
    slope = ema_now - ema_line.iloc[-1 - slope_lookback]

    if adx_val < sideways_adx:
        return SIDEWAYS
    if adx_val > trending_adx and price > ema_now and slope > 0:
        return BULL
    if adx_val > trending_adx and price < ema_now and slope < 0:
        return BEAR
    return SIDEWAYS


def strategy_allows_regime(strategy_name: str, regime: str) -> bool:
    """True if `strategy_name` is designed to trade in `regime`. Strategies
    not in STRATEGY_REGIMES are never gated (unknown = don't block)."""
    allowed = STRATEGY_REGIMES.get(strategy_name)
    if allowed is None:
        return True
    return regime in allowed
