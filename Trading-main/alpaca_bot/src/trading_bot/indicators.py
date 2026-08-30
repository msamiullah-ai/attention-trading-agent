"""Technical indicators computed from scratch with pandas/numpy — no TA-Lib.

Every function takes a DataFrame with lowercase OHLCV columns
(open, high, low, close, volume) and returns a Series or DataFrame aligned to
the same index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    return df[column].rolling(period).mean()


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    return df[column].ewm(span=period, adjust=False).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: seed with a plain mean of the first `period` valid
    values, then recurse as avg = (prev * (period - 1) + x) / period.

    A plain `.ewm(alpha=1/period, adjust=False)` looks similar but seeds from
    the first data point instead of a period-length average, which biases
    every value inside the seed window.
    """
    arr = series.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    valid = ~np.isnan(arr)
    if not valid.any():
        return pd.Series(out, index=series.index)

    first_valid = int(np.argmax(valid))
    if len(arr) - first_valid < period:
        return pd.Series(out, index=series.index)

    seed_pos = first_valid + period - 1
    prev = arr[first_valid:first_valid + period].mean()
    out[seed_pos] = prev
    for i in range(seed_pos + 1, len(arr)):
        prev = (prev * (period - 1) + arr[i]) / period
        out[i] = prev
    return pd.Series(out, index=series.index)


def rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """Wilder's RSI, seeded per `_wilder_smooth` (not a plain EMA)."""
    delta = df[column].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100 - (100 / (1 + rs))
    # Wilder RSI is 100 when there are only gains (avg_loss == 0).
    result = result.where(avg_loss != 0.0, 100.0)
    return result


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range. The first bar has no previous close, so it falls back to
    high - low (pandas' skipna default on the row-wise max) rather than NaN.
    """
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR: same smoothing as RSI, seeded by the first `period` bars."""
    return _wilder_smooth(true_range(df), period)


def vwap(df: pd.DataFrame, tz: str = "America/New_York") -> pd.Series:
    """Intraday VWAP that resets at the start of each calendar day in `tz`
    (index tz-aware). Default is NY market time, for equities; crypto never
    closes, so pass tz="UTC" there to reset at UTC midnight instead.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]

    day = df.index.tz_convert(tz).date if df.index.tz is not None else df.index.date
    day = pd.Index(day, name="day")

    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return (cum_pv / cum_vol).set_axis(df.index)


def bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = "close"
) -> pd.DataFrame:
    mid = sma(df, period, column)
    std = df[column].rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def bollinger_bandwidth(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = "close"
) -> pd.Series:
    bands = bollinger_bands(df, period, std_dev, column)
    return (bands["upper"] - bands["lower"]) / bands["mid"]


def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close"
) -> pd.DataFrame:
    fast_ema = ema(df, fast, column)
    slow_ema = ema(df, slow, column)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index: trend strength on a 0-100 scale,
    direction-agnostic (a strong downtrend scores as high as a strong uptrend).

    +DM_i = high_i - high_{i-1}, kept only if it exceeds the corresponding
    -DM and is positive (an "up move" bar); -DM is the mirror image using
    lows. Each is Wilder-smoothed like ATR, turned into a directional index
    (DI), and DX = 100 * |+DI - -DI| / (+DI + -DI) is itself Wilder-smoothed
    to produce ADX.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smoothed_tr = _wilder_smooth(true_range(df), period)
    smoothed_plus_dm = _wilder_smooth(plus_dm, period)
    smoothed_minus_dm = _wilder_smooth(minus_dm, period)

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0.0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return _wilder_smooth(dx, period)


def realized_volatility(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """Rolling standard deviation of log returns -- a standard crypto
    volatility measure since crypto has no fixed session to anchor an
    intraday range-based measure against. Not annualized; scale by the
    caller if needed (e.g. * sqrt(bars_per_year)).
    """
    log_returns = np.log(df[column] / df[column].shift(1))
    return log_returns.rolling(period).std()


def rolling_high(df: pd.DataFrame, period: int = 10, column: str = "close") -> pd.Series:
    """Rolling max, inclusive of the current bar -- so `price >= rolling_high`
    on the current bar means "today is a new N-period high"."""
    return df[column].rolling(period).max()


def rolling_low(df: pd.DataFrame, period: int = 10, column: str = "close") -> pd.Series:
    return df[column].rolling(period).min()


def zscore(series: pd.Series, period: int = 30) -> pd.Series:
    """How many rolling standard deviations `series` is from its own rolling
    mean. Used to flag "unusually stretched" readings, e.g. of a residual
    return series in a mean-reversion strategy."""
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_beta(y: pd.Series, x: pd.Series, period: int = 30) -> pd.Series:
    """Rolling regression beta of y's returns on x's returns:
    cov(x_ret, y_ret) / var(x_ret). E.g. an altcoin's beta to BTC, used to
    strip out the BTC-driven component of its return (the "residual").
    """
    x_ret = x.pct_change()
    y_ret = y.pct_change()
    cov = x_ret.rolling(period).cov(y_ret)
    var = x_ret.rolling(period).var()
    return cov / var.replace(0.0, np.nan)
