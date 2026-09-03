"""Signal generation: pluggable strategies with a shared interface.

Each strategy answers three questions from a bars DataFrame:
  - generate_signal(df)                 -> "BUY" | "SELL" | "HOLD"
  - get_stop_loss(entry_price, df, side) -> price
  - get_take_profit(entry_price, df, side) -> price

Strategies are position-agnostic: they read market data and cannot see
whether trader.py currently holds a position. trader.py decides how to
interpret BUY/SELL depending on what's already open (open new / close / flip).

Every indicator used here is strictly causal (rolling/ewm/cumulative — never
looks ahead), so `precompute(df)` computed once over a full history and read
bar-by-bar gives identical results to recomputing on each truncated window.
generate_signal() takes the simple (recompute-per-call) path, which is fine
live since it only runs once per poll; the backtest engine calls precompute()
once and reads from it per bar to avoid O(n^2) recomputation over history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, time, timedelta

import pandas as pd

from . import indicators as ind

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


class Strategy(ABC):
    name: str = "base"
    # Bars needed before generate_signal can produce anything but HOLD.
    min_bars: int = 30
    # Whether a SELL signal while flat may open a short. Long-only strategies
    # (e.g. EmaRsiStrategy) must set this False, since for them SELL only
    # ever means "exit an existing long", never "open a short".
    allow_short: bool = True

    def strength(self, df: pd.DataFrame) -> float:
        """How strong THIS signal is, 0..1. Neutral by default.

        Used only when position slots are scarce and something has to rank the
        candidates -- see `scheduling.DeficitRoundRobin`. It is not a
        probability and it is not comparable across strategies; it only has to
        order one strategy's own signals sensibly against each other.

        The neutral default is deliberate: a strategy with no meaningful notion
        of conviction should say so and let the scheduler's fairness term
        decide, rather than invent a number that looks like information.
        """
        return 0.5

    @abstractmethod
    def precompute(self, df: pd.DataFrame) -> dict:
        """Compute this strategy's indicators once over the full `df`."""

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        """BUY/SELL/HOLD at row `i`, reading from `ctx` (see `precompute`)."""

    def generate_signal(self, df: pd.DataFrame) -> str:
        """Return BUY/SELL/HOLD based on the most recent bar in `df`."""
        if len(df) < self.min_bars:
            return HOLD
        return self.evaluate(df, self.precompute(df), len(df) - 1)

    @abstractmethod
    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        """Stop-loss price for a position opened at `entry_price`. `side` is
        "long" or "short"."""

    @abstractmethod
    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        """Take-profit price for a position opened at `entry_price`."""


class EmaRsiStrategy(Strategy):
    """Trend-filtered oversold reversal, long only.

    Only takes trades when price is above EMA_20 (uptrend bias). Entry fires
    when RSI_14 was below 30 on the previous bar and crosses back above 30 on
    this bar. Position also exits when RSI climbs above 70.
    """

    name = "ema_rsi"
    min_bars = 21
    allow_short = False

    def __init__(
        self,
        ema_period: int = 20,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_period: int = 14,
        stop_atr_mult: float = 2.0,
        target_atr_mult: float = 4.0,
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.min_bars = max(ema_period, rsi_period, atr_period) + 1

    def precompute(self, df: pd.DataFrame) -> dict:
        return {"ema": ind.ema(df, self.ema_period), "rsi": ind.rsi(df, self.rsi_period)}

    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        ema_line, rsi_line = ctx["ema"], ctx["rsi"]
        price = df["close"].iloc[i]
        rsi_prev, rsi_now = rsi_line.iloc[i - 1], rsi_line.iloc[i]

        if rsi_now > self.rsi_overbought:
            return SELL

        crossed_up = rsi_prev < self.rsi_oversold <= rsi_now
        uptrend = price > ema_line.iloc[i]
        if crossed_up and uptrend:
            return BUY

        return HOLD

    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "EmaRsiStrategy is long-only (allow_short=False)"
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price - self.stop_atr_mult * atr_val

    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "EmaRsiStrategy is long-only (allow_short=False)"
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price + self.target_atr_mult * atr_val


class VwapMeanReversionStrategy(Strategy):
    """Fade extreme moves away from intraday VWAP, both directions.

    Long when price is far enough below VWAP and RSI confirms oversold;
    short the mirror image. Only trades mid-session (10:00-15:30 ET) to skip
    open/close volatility. Stop is a tight 1x ATR; target is VWAP itself.
    """

    name = "vwap_mean_reversion"
    min_bars = 21

    def __init__(
        self,
        atr_period: int = 14,
        rsi_period: int = 14,
        atr_band_mult: float = 1.5,
        stop_atr_mult: float = 1.0,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        session_start: time = time(10, 0),
        session_end: time = time(15, 30),
    ):
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.atr_band_mult = atr_band_mult
        self.stop_atr_mult = stop_atr_mult
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.session_start = session_start
        self.session_end = session_end
        self.min_bars = max(atr_period, rsi_period) + 1

    def _in_session(self, ts: pd.Timestamp) -> bool:
        local = ts.tz_convert("America/New_York") if ts.tzinfo is not None else ts
        return self.session_start <= local.time() <= self.session_end

    def precompute(self, df: pd.DataFrame) -> dict:
        return {
            "vwap": ind.vwap(df),
            "atr": ind.atr(df, self.atr_period),
            "rsi": ind.rsi(df, self.rsi_period),
        }

    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        if not self._in_session(df.index[i]):
            return HOLD

        price = df["close"].iloc[i]
        vwap_now = ctx["vwap"].iloc[i]
        band = self.atr_band_mult * ctx["atr"].iloc[i]
        rsi_now = ctx["rsi"].iloc[i]

        if price < vwap_now - band and rsi_now < self.rsi_oversold:
            return BUY
        if price > vwap_now + band and rsi_now > self.rsi_overbought:
            return SELL
        return HOLD

    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        offset = self.stop_atr_mult * atr_val
        return entry_price - offset if side == "long" else entry_price + offset

    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        # Mean-reversion target is VWAP itself, evaluated at entry time.
        return float(ind.vwap(df).iloc[-1])


class BollingerSqueezeStrategy(Strategy):
    """Volatility contraction followed by a confirmed directional breakout.

    A "squeeze" is bandwidth making a new 20-period low. The bar after a
    squeeze that closes outside the band, on above-average volume, triggers
    entry in the breakout direction.
    """

    name = "bollinger_squeeze"
    min_bars = 41

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        squeeze_lookback: int = 20,
        volume_period: int = 20,
        volume_mult: float = 1.5,
        target_width_mult: float = 2.0,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.volume_period = volume_period
        self.volume_mult = volume_mult
        self.target_width_mult = target_width_mult
        self.min_bars = bb_period + squeeze_lookback + 1

    def _bands_and_squeeze(self, df: pd.DataFrame):
        bands = ind.bollinger_bands(df, self.bb_period, self.bb_std)
        bandwidth = ind.bollinger_bandwidth(df, self.bb_period, self.bb_std)
        # New 20-period low, evaluated using only *prior* bars (exclude current).
        rolling_low = bandwidth.shift(1).rolling(self.squeeze_lookback).min()
        squeeze = bandwidth <= rolling_low
        return bands, squeeze

    def precompute(self, df: pd.DataFrame) -> dict:
        bands, squeeze = self._bands_and_squeeze(df)
        return {"bands": bands, "squeeze": squeeze, "vol_ma": ind.volume_ma(df, self.volume_period)}

    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        bands, squeeze, vol_ma = ctx["bands"], ctx["squeeze"], ctx["vol_ma"]

        # Require the squeeze on the prior bar; today confirms the breakout.
        was_squeezed = bool(squeeze.iloc[i - 1])
        volume_confirmed = df["volume"].iloc[i] > self.volume_mult * vol_ma.iloc[i]

        if not (was_squeezed and volume_confirmed):
            return HOLD

        price = df["close"].iloc[i]
        if price > bands["upper"].iloc[i]:
            return BUY
        if price < bands["lower"].iloc[i]:
            return SELL
        return HOLD

    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        bands, _ = self._bands_and_squeeze(df)
        return float(bands["lower"].iloc[-1] if side == "long" else bands["upper"].iloc[-1])

    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        bands, _ = self._bands_and_squeeze(df)
        width = bands["upper"].iloc[-1] - bands["lower"].iloc[-1]
        offset = self.target_width_mult * width
        return entry_price + offset if side == "long" else entry_price - offset


class CryptoMomentumStrategy(Strategy):
    """Breakout momentum with a volatility filter, sized for crypto's fatter
    tails than equities (wider ATR multiples than the equity strategies use).

    Entry requires four things on the same bar: price closes at a new
    N-period high; realized volatility is calm relative to its own recent
    history (at or below its trailing 80th percentile -- avoids buying into
    erratic, whipsaw conditions); volume confirms (above its 20-period
    average); and RSI isn't already extreme (<=80, avoids chasing a
    blow-off top). Exit fires on a new N-period low, mirroring entry -- the
    ATR stop/target below is the primary protection, this is secondary.

    Long-only: Alpaca's crypto is spot only, no margin/borrow, so shorting
    is rejected outright (confirmed against the live API -- "insufficient
    balance"). Two things from the original design aren't implemented here
    since they don't fit a single-symbol, stateless evaluate(): liquidity
    filtering by market cap (that's a symbol-selection concern for
    config.yaml, not strategy logic) and reducing altcoin size when BTC is
    in a downtrend (would need cross-symbol context this interface doesn't
    have access to).
    """

    name = "crypto_momentum"
    allow_short = False

    def __init__(
        self,
        high_low_period: int = 10,
        vol_period: int = 20,
        vol_percentile_window: int = 100,
        vol_percentile: float = 0.80,
        rsi_period: int = 14,
        rsi_ceiling: float = 80.0,
        volume_period: int = 20,
        atr_period: int = 14,
        stop_atr_mult: float = 2.5,
        target_atr_mult: float = 5.0,
    ):
        self.high_low_period = high_low_period
        self.vol_period = vol_period
        self.vol_percentile_window = vol_percentile_window
        self.vol_percentile = vol_percentile
        self.rsi_period = rsi_period
        self.rsi_ceiling = rsi_ceiling
        self.volume_period = volume_period
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.min_bars = vol_percentile_window + vol_period + 2

    def precompute(self, df: pd.DataFrame) -> dict:
        vol = ind.realized_volatility(df, self.vol_period)
        vol_threshold = vol.rolling(self.vol_percentile_window).quantile(self.vol_percentile)
        return {
            "high": ind.rolling_high(df, self.high_low_period),
            "low": ind.rolling_low(df, self.high_low_period),
            "vol": vol,
            "vol_threshold": vol_threshold,
            "rsi": ind.rsi(df, self.rsi_period),
            "vol_ma": ind.volume_ma(df, self.volume_period),
        }

    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        price = df["close"].iloc[i]
        new_high = price >= ctx["high"].iloc[i]
        new_low = price <= ctx["low"].iloc[i]
        calm = ctx["vol"].iloc[i] <= ctx["vol_threshold"].iloc[i]
        volume_confirmed = df["volume"].iloc[i] > ctx["vol_ma"].iloc[i]
        rsi_ok = ctx["rsi"].iloc[i] <= self.rsi_ceiling

        if new_high and calm and volume_confirmed and rsi_ok:
            return BUY
        if new_low:
            return SELL
        return HOLD

    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "CryptoMomentumStrategy is long-only (allow_short=False)"
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price - self.stop_atr_mult * atr_val

    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "CryptoMomentumStrategy is long-only (allow_short=False)"
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price + self.target_atr_mult * atr_val



class OpeningRangeBreakoutStrategy(Strategy):
    """Break of the first N minutes' range, volume-confirmed, stop at the range.

    WHY THIS ONE

    Opening Range Breakout is one of the few intraday equity patterns with a
    published, replicated result rather than a folk reputation: Zarattini &
    Aziz, "Can Day Trading Really Be Profitable?" (2023), tested a short opening
    range on liquid US equities across two decades. The edge documented there is
    not the breakout itself -- it is the combination of an early objective
    level, a hard stop at the opposite side of that level, and no overnight
    exposure.

    WHY IT SUITS THIS BOT

    The four strategies above all fire on indicator crossings, which on 1-minute
    bars means often, and mostly on noise. Measured on this account's own data:
    vwap_mean_reversion signalled on ~15% of bars, bollinger_squeeze on 0.2%,
    and each showed negative expectancy over its first twenty trades before the
    risk layer paused it.

    ORB fires at most twice per symbol per day, at a level fixed before any
    signal exists. Fewer decisions, each with a defined invalidation point, is
    the structural difference -- not another oscillator.

    THE RULES, AND WHY EACH IS THERE

      range   high/low of the first `range_minutes` after the open, frozen once
              the window closes. A level that keeps moving is not a level.
      entry   a close beyond the range, in the direction of the break.
      volume  the breaking bar must trade above its own recent average. An
              unconfirmed break is the signature of a false one -- price pokes
              through on nothing and reverts -- and it is the dominant ORB
              failure mode, so a version without this filter bleeds.
      once    one entry per side per day. Re-entering a level that already
              failed is how one choppy session becomes ten losses.
      cutoff  no new entries after `no_entry_after`: a breakout with twenty
              minutes left cannot reach a sensible target.
      stop    the opposite side of the range. Deliberately NOT an ATR multiple
              -- the range is the thesis, so if price returns through it the
              reason for the trade is gone. It also makes R the range width,
              known at entry rather than inferred afterwards.
      target  a multiple of that same R.

    Both directions trade: a downside break is the same structure mirrored.

    WHAT IS NOT CLAIMED: that it is profitable here. It is better specified than
    the four above, on a timeframe it was designed for, and that is all that can
    be said until it is backtested on this account's data.
    """

    """Break of the first N minutes' range, volume-confirmed, stop at the range."""

    name = "opening_range_breakout"
    min_bars = 30
    allow_short = True

    def __init__(
        self,
        range_minutes: int = 15,
        volume_period: int = 20,
        volume_mult: float = 1.2,
        target_r: float = 2.0,
        session_open: time = time(9, 30),
        no_entry_after: time = time(15, 0),
    ):
        self.range_minutes = range_minutes
        self.volume_period = volume_period
        self.volume_mult = volume_mult
        self.target_r = target_r
        self.session_open = session_open
        self.no_entry_after = no_entry_after
        self.min_bars = max(range_minutes, volume_period) + 2

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _local(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_convert("America/New_York") if ts.tzinfo is not None else ts

    def _ranges(self, df: pd.DataFrame) -> dict:
        """Each session's opening range, keyed by local trading date.

        Grouping on the LOCAL date is load-bearing: a UTC-day grouping splits
        the US session across two buckets and yields a range that never existed.
        """
        local = [self._local(t) for t in df.index]
        dates = pd.Series([t.date() for t in local], index=df.index)
        times = pd.Series([t.time() for t in local], index=df.index)

        out: dict = {}
        for day, day_df in df.groupby(dates, sort=False):
            day_times = times.loc[day_df.index]
            end = (datetime.combine(day, self.session_open)
                   + timedelta(minutes=self.range_minutes)).time()
            window = day_df[(day_times >= self.session_open) & (day_times < end)]
            if len(window) < 2:
                continue          # holiday, half day, or missing bars
            out[day] = (float(window["high"].max()), float(window["low"].min()), end)
        return {"ranges": out, "dates": dates, "times": times}

    # -- Strategy interface --------------------------------------------------

    def precompute(self, df: pd.DataFrame) -> dict:
        ctx = self._ranges(df)
        ctx["vol_ma"] = ind.volume_ma(df, self.volume_period)
        # One entry per side per day, tracked in the context rather than on
        # self so a backtest and a live run can never contaminate each other.
        ctx["taken"] = set()
        return ctx

    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        day = ctx["dates"].iloc[i]
        rng = ctx["ranges"].get(day)
        if rng is None:
            return HOLD
        high, low, range_end = rng

        now = ctx["times"].iloc[i]
        if now < range_end:
            return HOLD                    # range still forming
        if now >= self.no_entry_after:
            return HOLD                    # too late to reach a target

        avg = ctx["vol_ma"].iloc[i]
        if pd.isna(avg) or avg <= 0:
            return HOLD
        if float(df["volume"].iloc[i]) < float(avg) * self.volume_mult:
            return HOLD                    # unconfirmed break

        close = float(df["close"].iloc[i])
        if close > high and (day, "long") not in ctx["taken"]:
            ctx["taken"].add((day, "long"))
            return BUY
        if close < low and (day, "short") not in ctx["taken"]:
            ctx["taken"].add((day, "short"))
            return SELL
        return HOLD

    def _range_for_last_bar(self, df: pd.DataFrame) -> tuple[float, float] | None:
        built = self._ranges(df)
        day = self._local(df.index[-1]).date()
        rng = built["ranges"].get(day)
        return (rng[0], rng[1]) if rng else None

    def strength(self, df: pd.DataFrame) -> float:
        """Conviction from how decisively the range broke, and on what volume.

        Two things separate a break worth taking from one that reverts, and
        both are already measured here:

          extension  how far beyond the level price closed, as a fraction of
                     the range width. A close 40% of a range beyond it is a
                     different event from one a tick past, and treating them
                     alike is what makes a breakout strategy look random.
          volume     the breaking bar's volume against its own recent average,
                     which is the filter that already gates entry -- this reads
                     the same number as a degree rather than a yes/no.

        Multiplied, not averaged: a decisive break on no volume and a marginal
        break on huge volume are both weak, and averaging would rate them
        middling instead.
        """
        rng = self._range_for_last_bar(df)
        if rng is None:
            return 0.5
        high, low = rng
        width = high - low
        if width <= 0:
            return 0.5
        close = float(df["close"].iloc[-1])
        beyond = (close - high) if close > high else (low - close) if close < low else 0.0
        extension = min(1.0, max(0.0, beyond / width))

        avg = ind.volume_ma(df, self.volume_period).iloc[-1]
        if pd.isna(avg) or avg <= 0:
            confirmation = 0.5
        else:
            # 1x average -> 0.0, 3x -> 1.0. Above 3x adds nothing: past a point
            # heavy volume stops being confirmation and starts being a crowd.
            ratio = float(df["volume"].iloc[-1]) / float(avg)
            confirmation = min(1.0, max(0.0, (ratio - 1.0) / 2.0))

        # Floor at 0.05 so a qualifying signal never scores zero -- it passed
        # every gate to get here, and zero would let the deficit alone rank it.
        return max(0.05, extension * confirmation)

    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        rng = self._range_for_last_bar(df)
        if rng is None:
            # No range formed. A fixed fraction rather than something that would
            # silently disable the stop.
            return entry_price * (0.99 if side == "long" else 1.01)
        high, low = rng
        stop = low if side == "long" else high
        # A stop on the wrong side of entry is not a stop. Can happen when the
        # break is evaluated on a bar that has already reverted through the
        # range; fall back rather than submit an order Alpaca would reject.
        if (side == "long" and stop >= entry_price) or (side == "short" and stop <= entry_price):
            return entry_price * (0.99 if side == "long" else 1.01)
        return stop

    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        stop = self.get_stop_loss(entry_price, df, side)
        risk = abs(entry_price - stop)
        if risk <= 0:
            return entry_price * (1.01 if side == "long" else 0.99)
        return (entry_price + self.target_r * risk if side == "long"
                else entry_price - self.target_r * risk)


REGISTRY: dict[str, type[Strategy]] = {
    EmaRsiStrategy.name: EmaRsiStrategy,
    VwapMeanReversionStrategy.name: VwapMeanReversionStrategy,
    BollingerSqueezeStrategy.name: BollingerSqueezeStrategy,
    CryptoMomentumStrategy.name: CryptoMomentumStrategy,
    OpeningRangeBreakoutStrategy.name: OpeningRangeBreakoutStrategy,
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))