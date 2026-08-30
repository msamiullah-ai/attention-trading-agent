"""Experimental strategy variants. Kept entirely separate from signals.py --
not registered in signals.REGISTRY, not importable via build_strategy(), and
nothing here modifies any existing file. Use these classes directly (see
scripts/backtest_loose_ema_rsi.py) rather than through the normal
config.yaml `strategy:` name lookup.
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .signals import BUY, HOLD, SELL, Strategy


class LooseEmaRsiStrategy(Strategy):
    """A deliberately looser variant of EmaRsiStrategy (signals.py), built
    to compare against it. The original almost never fires -- its own
    1-year backtest found ~14 entries/year on a volatile stock and 0 on
    several calmer ones -- because it requires two rare conditions to land
    on the exact same bar. Two things are relaxed here:

    1. No same-bar cross requirement. The original needs RSI to cross the
       oversold threshold on *this specific bar*; if the dip happened two
       bars ago and the bounce is happening now, the original already
       missed it. This version fires if RSI touched `rsi_oversold` at any
       point in the last `lookback_bars` bars and is now turning back up
       (rsi_now > rsi_prev) -- a window instead of an instant.
    2. The EMA uptrend filter is optional (`require_uptrend`, default
       False) instead of mandatory. The original needs both the RSI
       pattern *and* price above its EMA20 on the same bar; making the
       second condition optional means the first alone is enough to fire.

    Unvalidated, on purpose -- this exists to be backtested and compared,
    not assumed better. Firing more often is not the same as firing more
    profitably; a looser trigger can just as easily be more frequently
    wrong. Also not registered with regime.py's STRATEGY_REGIMES map, so
    (unlike the live ema_rsi bot) nothing here gets blocked by the regime
    gate -- another reason results from this need to be checked, not trusted.
    """

    name = "loose_ema_rsi"
    allow_short = False

    def __init__(
        self,
        ema_period: int = 20,
        rsi_period: int = 14,
        rsi_oversold: float = 40.0,
        rsi_overbought: float = 70.0,
        lookback_bars: int = 5,
        require_uptrend: bool = False,
        atr_period: int = 14,
        stop_atr_mult: float = 2.0,
        target_atr_mult: float = 4.0,
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.lookback_bars = lookback_bars
        self.require_uptrend = require_uptrend
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.min_bars = max(ema_period, rsi_period, atr_period, lookback_bars) + 1

    def precompute(self, df: pd.DataFrame) -> dict:
        return {"ema": ind.ema(df, self.ema_period), "rsi": ind.rsi(df, self.rsi_period)}

    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        ema_line, rsi_line = ctx["ema"], ctx["rsi"]
        price = df["close"].iloc[i]
        rsi_now = rsi_line.iloc[i]
        rsi_prev = rsi_line.iloc[i - 1]

        if rsi_now > self.rsi_overbought:
            return SELL

        window_start = max(0, i - self.lookback_bars)
        was_oversold_recently = bool((rsi_line.iloc[window_start:i] <= self.rsi_oversold).any())
        turning_up = rsi_now > rsi_prev
        uptrend_ok = (not self.require_uptrend) or price > ema_line.iloc[i]

        if was_oversold_recently and turning_up and uptrend_ok:
            return BUY

        return HOLD

    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "LooseEmaRsiStrategy is long-only (allow_short=False)"
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price - self.stop_atr_mult * atr_val

    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "LooseEmaRsiStrategy is long-only (allow_short=False)"
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price + self.target_atr_mult * atr_val
