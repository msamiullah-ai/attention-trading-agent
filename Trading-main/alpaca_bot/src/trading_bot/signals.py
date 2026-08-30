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
from datetime import time

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


class BtcRelativeMeanReversionStrategy(Strategy):
    """Mean-reversion on an altcoin's return *relative to BTC* -- its
    "residual" return after subtracting the BTC-driven component
    (beta * btc_return), isolating the coin's own idiosyncratic move.
    Long only; deliberately NOT market-neutral -- see below.

    Entry: the residual return's rolling z-score drops to/below `entry_z`
    (the coin is unusually cheap relative to how BTC alone would explain its
    move). Exit fires on either side of the thesis: z climbing back to 0
    (reverted to the mean -- take profit) or z falling to/below `stop_z`
    (still falling -- the mean-reversion thesis has failed). Both are
    returned as SELL from evaluate(), same pattern as EmaRsiStrategy's
    signal-driven exit; get_stop_loss/get_take_profit below are a plain ATR
    backstop against a violent single-bar move before the next evaluation
    catches the z-score breakdown, not the primary exit mechanism.

    Two real limitations versus the design this is adapted from:

    1. NOT hedged. The original design shorts $beta worth of BTC perpetual
       per $1 of the long to stay market-neutral. Alpaca's crypto is spot
       only -- no perpetuals, no margin/borrow -- so shorting is rejected
       outright (confirmed against the live API: "insufficient balance").
       This is therefore a plain directional long, fully exposed to BTC's
       own moves, not a market-neutral pairs trade. Size accordingly.

    2. NOT in the strategy REGISTRY / not selectable via config.yaml's
       `strategy:` field. Every other strategy here gets one symbol's
       DataFrame per evaluate() call; this one inherently needs a second
       symbol's data (BTC) simultaneously, which that single-df interface
       doesn't carry. Rather than bolt a second data-fetch onto the
       Strategy contract for every strategy, this one takes its BTC
       reference series once, in the constructor -- so live use means
       refreshing it and reconstructing the strategy yourself each cycle;
       it does not re-fetch BTC on its own. Fine for backtesting (fetch
       both symbols' bars once, pass BTC's in); construct manually rather
       than via build_strategy().
    """

    name = "btc_relative_mean_reversion"
    allow_short = False

    def __init__(
        self,
        btc_df: pd.DataFrame,
        beta_period: int = 30,
        zscore_period: int = 30,
        entry_z: float = -2.0,
        stop_z: float = -3.0,
        atr_period: int = 14,
        stop_atr_mult: float = 2.0,
    ):
        self.btc_df = btc_df
        self.beta_period = beta_period
        self.zscore_period = zscore_period
        self.entry_z = entry_z
        self.stop_z = stop_z
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.min_bars = max(beta_period, zscore_period, atr_period) + 2

    def precompute(self, df: pd.DataFrame) -> dict:
        btc_close = self.btc_df["close"].reindex(df.index).ffill()
        beta = ind.rolling_beta(df["close"], btc_close, period=self.beta_period)
        alt_return = df["close"].pct_change()
        btc_return = btc_close.pct_change()
        residual = alt_return - beta * btc_return
        return {"z": ind.zscore(residual, self.zscore_period)}

    def evaluate(self, df: pd.DataFrame, ctx: dict, i: int) -> str:
        z = ctx["z"].iloc[i]
        if pd.isna(z):
            return HOLD
        if z <= self.stop_z or z >= 0:
            return SELL  # breakdown (thesis failed) or reverted (take profit)
        if z <= self.entry_z:
            return BUY
        return HOLD

    def get_stop_loss(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "BtcRelativeMeanReversionStrategy is long-only (allow_short=False)"
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price - self.stop_atr_mult * atr_val

    def get_take_profit(self, entry_price: float, df: pd.DataFrame, side: str) -> float:
        assert side == "long", "BtcRelativeMeanReversionStrategy is long-only (allow_short=False)"
        # No fixed price target -- the real exit is the z-score reverting to
        # 0 (handled in evaluate). This is a generous backstop only.
        atr_val = ind.atr(df, self.atr_period).iloc[-1]
        return entry_price + 4 * self.stop_atr_mult * atr_val


REGISTRY: dict[str, type[Strategy]] = {
    EmaRsiStrategy.name: EmaRsiStrategy,
    VwapMeanReversionStrategy.name: VwapMeanReversionStrategy,
    BollingerSqueezeStrategy.name: BollingerSqueezeStrategy,
    CryptoMomentumStrategy.name: CryptoMomentumStrategy,
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**(params or {}))
