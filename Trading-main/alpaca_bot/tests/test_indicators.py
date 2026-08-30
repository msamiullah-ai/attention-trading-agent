"""Indicator tests against hand-computed values (no network access)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot import indicators as ind  # noqa: E402


def make_df(closes, highs=None, lows=None, volumes=None, freq="1min", tz="UTC") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq=freq, tz=tz)
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c + 0.5 for c in closes],
            "low": lows if lows is not None else [c - 0.5 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000] * n,
        },
        index=idx,
    )


def test_sma_matches_manual_average():
    df = make_df([1, 2, 3, 4, 5])
    result = ind.sma(df, period=3)
    assert result.iloc[-1] == pytest.approx((3 + 4 + 5) / 3)
    assert pd.isna(result.iloc[0])


def test_ema_matches_manual_recursion():
    df = make_df([10, 12, 14, 11, 13])
    period = 3
    alpha = 2 / (period + 1)
    manual = df["close"].iloc[0]
    for price in df["close"].iloc[1:]:
        manual = alpha * price + (1 - alpha) * manual
    result = ind.ema(df, period=period)
    assert result.iloc[-1] == pytest.approx(manual)


def test_rsi_is_100_when_only_gains():
    df = make_df(list(range(1, 20)))  # strictly increasing -> no losses
    result = ind.rsi(df, period=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_only_losses():
    df = make_df(list(range(20, 1, -1)))  # strictly decreasing -> no gains
    result = ind.rsi(df, period=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_classic_wilder_example():
    # Classic Wilder RSI worked example (14-period), values from his original
    # illustration. Expect RSI near 70 after the seed period.
    closes = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28,
    ]
    df = make_df(closes)
    result = ind.rsi(df, period=14)
    assert result.iloc[-1] == pytest.approx(70.5, abs=1.0)


def test_true_range_first_bar_falls_back_to_high_minus_low():
    df = make_df([10], highs=[10.5], lows=[9.5])
    assert ind.true_range(df).iloc[0] == pytest.approx(1.0)


def test_atr_matches_manual_true_range_average():
    df = make_df([10, 11, 10.5, 12, 11.5], highs=[10.5, 11.5, 11, 12.5, 12], lows=[9.5, 10.5, 10, 11.5, 11])
    result = ind.atr(df, period=3)
    tr = ind.true_range(df)

    # Wilder ATR seeds with a plain average of the first `period` true ranges...
    seed = tr.iloc[0:3].mean()
    assert result.iloc[2] == pytest.approx(seed, rel=1e-6)
    # ...then recurses: avg = (prev * (period - 1) + current) / period.
    expected_next = (seed * 2 + tr.iloc[3]) / 3
    assert result.iloc[3] == pytest.approx(expected_next, rel=1e-6)


def test_bollinger_bands_width_scales_with_std_dev():
    df = make_df([10, 12, 9, 11, 13, 8, 14, 10, 12, 9, 11, 13, 8, 14, 10, 12, 9, 11, 13, 8])
    bands_2 = ind.bollinger_bands(df, period=20, std_dev=2.0)
    bands_1 = ind.bollinger_bands(df, period=20, std_dev=1.0)
    width_2 = bands_2["upper"].iloc[-1] - bands_2["lower"].iloc[-1]
    width_1 = bands_1["upper"].iloc[-1] - bands_1["lower"].iloc[-1]
    assert width_2 == pytest.approx(2 * width_1)


def test_bollinger_bandwidth_is_relative_width():
    df = make_df([10, 12, 9, 11, 13, 8, 14, 10, 12, 9, 11, 13, 8, 14, 10, 12, 9, 11, 13, 8])
    bands = ind.bollinger_bands(df, period=20, std_dev=2.0)
    bandwidth = ind.bollinger_bandwidth(df, period=20, std_dev=2.0)
    expected = (bands["upper"].iloc[-1] - bands["lower"].iloc[-1]) / bands["mid"].iloc[-1]
    assert bandwidth.iloc[-1] == pytest.approx(expected)


def test_macd_histogram_is_macd_minus_signal():
    df = make_df(list(np.linspace(100, 130, 60)))
    result = ind.macd(df, fast=12, slow=26, signal=9)
    assert result["histogram"].iloc[-1] == pytest.approx(
        result["macd"].iloc[-1] - result["signal"].iloc[-1]
    )


def test_volume_ma_matches_manual_average():
    df = make_df([1] * 5, volumes=[100, 200, 300, 400, 500])
    result = ind.volume_ma(df, period=3)
    assert result.iloc[-1] == pytest.approx((300 + 400 + 500) / 3)


def test_vwap_resets_each_day():
    # Two days of two bars each, constant price/volume -> VWAP == price all day.
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 09:30", "2024-01-02 09:31",
            "2024-01-03 09:30", "2024-01-03 09:31",
        ],
        tz="UTC",
    )
    df = pd.DataFrame(
        {
            "open": [10, 20, 30, 40],
            "high": [10, 20, 30, 40],
            "low": [10, 20, 30, 40],
            "close": [10, 20, 30, 40],
            "volume": [100, 100, 100, 100],
        },
        index=idx,
    )
    result = ind.vwap(df)
    # Day 1: cumulative average of 10 then (10+20)/2=15.
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx(15.0)
    # Day 2 resets, independent of day 1.
    assert result.iloc[2] == pytest.approx(30.0)
    assert result.iloc[3] == pytest.approx(35.0)


def test_vwap_resets_at_utc_midnight_when_requested():
    # Bars straddle NY midnight but not UTC midnight -> tz="UTC" must NOT reset here.
    idx = pd.DatetimeIndex(
        ["2024-01-02 00:30", "2024-01-02 01:30", "2024-01-02 12:00", "2024-01-02 13:00"], tz="UTC"
    )
    df = pd.DataFrame(
        {"open": [10, 20, 30, 40], "high": [10, 20, 30, 40], "low": [10, 20, 30, 40],
         "close": [10, 20, 30, 40], "volume": [100] * 4},
        index=idx,
    )
    utc_vwap = ind.vwap(df, tz="UTC")
    ny_vwap = ind.vwap(df, tz="America/New_York")
    # No reset under UTC -> plain cumulative average across all 4 bars.
    assert utc_vwap.tolist() == pytest.approx([10.0, 15.0, 20.0, 25.0])
    # NY resets between bar 2 and 3 (00:30/01:30 UTC are still 01-01 in NY).
    assert ny_vwap.tolist() == pytest.approx([10.0, 15.0, 30.0, 35.0])


def test_adx_is_high_for_a_pure_trend():
    df = make_df(list(np.linspace(100, 200, 60)))  # monotonic, no down-moves at all
    result = ind.adx(df, period=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_adx_is_low_for_a_choppy_series():
    df = make_df([100 + (5 if i % 2 == 0 else -5) for i in range(60)])
    result = ind.adx(df, period=14)
    assert result.iloc[-1] < 20.0


def test_realized_volatility_matches_manual_log_return_std():
    df = make_df(list(np.linspace(100, 110, 30)))
    result = ind.realized_volatility(df, period=20)
    manual = np.log(df["close"] / df["close"].shift(1)).rolling(20).std()
    pd.testing.assert_series_equal(result, manual, check_names=False)


def test_rolling_high_low_track_the_window():
    df = make_df([100, 105, 103, 108, 104, 102, 110, 106])
    high = ind.rolling_high(df, period=3)
    low = ind.rolling_low(df, period=3)
    assert high.iloc[3] == pytest.approx(108.0)  # max of [105,103,108]
    assert low.iloc[3] == pytest.approx(103.0)   # min of [105,103,108]
    assert high.iloc[6] == pytest.approx(110.0)  # max of [104,102,110]


def test_zscore_flags_an_outlier_positive():
    series = pd.Series([10.0, 12.0, 11.0, 13.0, 50.0, 12.0, 11.0])
    result = ind.zscore(series, period=4)
    assert result.iloc[4] > 1.0  # the spike, relative to its own trailing window


def test_zscore_is_roughly_zero_for_a_flat_series():
    series = pd.Series([10.0] * 10)
    result = ind.zscore(series, period=4)
    assert result.iloc[-1] != result.iloc[-1] or abs(result.iloc[-1]) < 1e-9  # NaN (zero std) or ~0


def test_rolling_beta_recovers_the_true_linear_relationship():
    rng = np.random.RandomState(0)
    x_ret = rng.normal(0, 0.01, 60)
    y_ret = 2 * x_ret  # y's returns are exactly 2x x's -> true beta = 2
    x = pd.Series(100 * np.cumprod(1 + x_ret))
    y = pd.Series(100 * np.cumprod(1 + y_ret))
    result = ind.rolling_beta(y, x, period=30)
    assert result.iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_rolling_beta_of_a_series_with_itself_is_one():
    df = make_df(list(np.linspace(100, 130, 40)))
    result = ind.rolling_beta(df["close"], df["close"], period=20)
    assert result.iloc[-1] == pytest.approx(1.0)
