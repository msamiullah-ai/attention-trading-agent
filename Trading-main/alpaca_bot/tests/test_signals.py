"""Strategy tests using synthetic bars (no network access).

Series were verified empirically against the actual indicator math (Wilder
RSI has recursive memory since inception, so hand-derived RSI values for a
short pullback are unreliable — these series were searched to actually cross
the thresholds, not guessed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.signals import (  # noqa: E402
    BUY,
    HOLD,
    SELL,
    BollingerSqueezeStrategy,
    BtcRelativeMeanReversionStrategy,
    CryptoMomentumStrategy,
    EmaRsiStrategy,
    VwapMeanReversionStrategy,
    build_strategy,
)


def make_df(closes, highs=None, lows=None, volumes=None, start="2024-01-02 09:30") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c + 0.2 for c in closes],
            "low": lows if lows is not None else [c - 0.2 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000] * n,
        },
        index=idx,
    )


# ---------------------------------------------------------------- EMA + RSI

def test_registry_builds_each_strategy():
    for name in ("ema_rsi", "vwap_mean_reversion", "bollinger_squeeze", "crypto_momentum"):
        assert build_strategy(name).name == name


def test_ema_rsi_holds_with_too_few_bars():
    strat = EmaRsiStrategy()
    df = make_df([100.0] * 5)
    assert strat.generate_signal(df) == HOLD


def test_ema_rsi_buys_on_oversold_reversal_in_uptrend():
    # Verified via search: RSI crosses 8.9 -> 52.8 while price (96.6) > EMA20 (93.1).
    up = list(np.linspace(100, 100 + 2 * 0.8, 2))
    decline = list(np.linspace(up[-1], up[-1] - 15, 20))[1:]
    bounce = [decline[-1] + 10]
    df = make_df(up + decline + bounce)
    strat = EmaRsiStrategy()
    assert strat.generate_signal(df) == BUY


def test_ema_rsi_sells_when_overbought():
    strat = EmaRsiStrategy()
    df = make_df(list(np.linspace(100, 140, 22)))  # monotonic rally -> RSI pinned high
    assert strat.generate_signal(df) == SELL


def test_ema_rsi_stop_and_target_use_atr_multiples():
    strat = EmaRsiStrategy(stop_atr_mult=2.0, target_atr_mult=4.0)
    df = make_df(list(np.linspace(100, 120, 30)))
    entry = 110.0
    stop = strat.get_stop_loss(entry, df, "long")
    target = strat.get_take_profit(entry, df, "long")
    assert stop < entry < target
    # 2:1 reward:risk by construction (4x ATR reward vs 2x ATR risk).
    assert (target - entry) == pytest.approx(2 * (entry - stop))


# -------------------------------------------------------------- VWAP fade

def test_vwap_reversion_holds_outside_session():
    strat = VwapMeanReversionStrategy()
    # 09:30 UTC lands well before 10:00 ET even accounting for offset.
    closes = [100.0] * 20 + [100.0, 100.0, 90.0]
    df = make_df(closes, start="2024-01-02 04:00")
    assert strat.generate_signal(df) == HOLD


def test_vwap_reversion_buys_when_price_drops_far_below_vwap():
    closes = [100.0] * 20 + [100.0, 100.0, 95.0]
    df = make_df(closes, start="2024-01-02 16:00")  # UTC 16:00 -> ~11:00 America/New_York
    strat = VwapMeanReversionStrategy()
    assert strat.generate_signal(df) == BUY


def test_vwap_reversion_sells_when_price_rises_far_above_vwap():
    closes = [100.0] * 20 + [100.0, 100.0, 105.0]
    df = make_df(closes, start="2024-01-02 16:00")
    strat = VwapMeanReversionStrategy()
    assert strat.generate_signal(df) == SELL


def test_vwap_reversion_target_is_vwap_at_entry():
    closes = [100.0] * 20 + [100.0, 100.0, 95.0]
    df = make_df(closes, start="2024-01-02 16:00")
    strat = VwapMeanReversionStrategy()
    target = strat.get_take_profit(95.0, df, "long")
    assert target == pytest.approx(99.78, abs=0.1)


def test_vwap_reversion_stop_direction_depends_on_side():
    df = make_df([100.0] * 25, start="2024-01-02 16:00")
    strat = VwapMeanReversionStrategy()
    long_stop = strat.get_stop_loss(100.0, df, "long")
    short_stop = strat.get_stop_loss(100.0, df, "short")
    assert long_stop < 100.0 < short_stop


# ---------------------------------------------------------- Bollinger squeeze

def test_bollinger_squeeze_buys_on_upside_breakout_with_volume():
    flat = [100 + (i % 2) * 0.05 for i in range(45)]
    closes = flat[:44] + [106.0]
    volumes = [1000] * 44 + [3000]
    df = make_df(closes, volumes=volumes)
    strat = BollingerSqueezeStrategy()
    assert strat.generate_signal(df) == BUY


def test_bollinger_squeeze_sells_on_downside_breakout_with_volume():
    flat = [100 + (i % 2) * 0.05 for i in range(45)]
    closes = flat[:44] + [94.0]
    volumes = [1000] * 44 + [3000]
    df = make_df(closes, volumes=volumes)
    strat = BollingerSqueezeStrategy()
    assert strat.generate_signal(df) == SELL


def test_bollinger_squeeze_holds_without_volume_confirmation():
    flat = [100 + (i % 2) * 0.05 for i in range(45)]
    closes = flat[:44] + [106.0]
    df = make_df(closes)  # default flat volume, no spike
    strat = BollingerSqueezeStrategy()
    assert strat.generate_signal(df) == HOLD


def test_bollinger_squeeze_stop_is_opposite_band():
    flat = [100 + (i % 2) * 0.05 for i in range(45)]
    closes = flat[:44] + [106.0]
    volumes = [1000] * 44 + [3000]
    df = make_df(closes, volumes=volumes)
    strat = BollingerSqueezeStrategy()
    long_stop = strat.get_stop_loss(106.0, df, "long")
    long_target = strat.get_take_profit(106.0, df, "long")
    assert long_stop < 106.0 < long_target


# ------------------------------------------------------------ crypto momentum

def _crypto_buy_series(strat: CryptoMomentumStrategy) -> list[float]:
    # Calm low-noise baseline, then a gentle ramp to a new high that stays
    # gentle enough per-bar to keep both realized vol and RSI from spiking.
    rng = np.random.RandomState(1)
    calm = 100 + rng.normal(0, 0.03, strat.min_bars - 10)
    ramp = calm[-1] * (1 + np.linspace(0.0002, 0.0015, 10))
    return list(calm) + list(ramp)


def test_crypto_momentum_holds_with_too_few_bars():
    strat = CryptoMomentumStrategy()
    df = make_df([100.0] * 10)
    assert strat.generate_signal(df) == HOLD


def test_crypto_momentum_is_long_only():
    assert CryptoMomentumStrategy().allow_short is False


def test_crypto_momentum_buys_on_calm_breakout_with_volume():
    strat = CryptoMomentumStrategy()
    closes = _crypto_buy_series(strat)
    volumes = [1000] * (len(closes) - 1) + [3000]
    df = make_df(closes, volumes=volumes)
    assert strat.generate_signal(df) == BUY


def test_crypto_momentum_holds_without_volume_confirmation():
    strat = CryptoMomentumStrategy()
    closes = _crypto_buy_series(strat)
    df = make_df(closes)  # flat default volume, no spike
    assert strat.generate_signal(df) == HOLD


def test_crypto_momentum_holds_when_volatility_is_not_calm():
    strat = CryptoMomentumStrategy()
    closes = _crypto_buy_series(strat)  # same calm-then-new-high series that normally buys...
    # ...except with a sharp dip+overshoot blip injected inside the recent vol
    # window, so current realized vol is no longer calm even though price
    # still reaches a genuine new high (confirmed empirically: without the
    # blip this series returns BUY; the blip alone flips it to HOLD).
    blip = len(closes) - 15
    closes[blip] *= 0.95
    closes[blip + 1] *= 1.06
    closes[-10:] = [max(closes[:-10]) * (1 + 0.001 * k) for k in range(1, 11)]

    volumes = [1000] * (len(closes) - 1) + [3000]
    df = make_df(closes, volumes=volumes)
    assert strat.generate_signal(df) == HOLD


def test_crypto_momentum_sells_on_new_low():
    strat = CryptoMomentumStrategy()
    closes = _crypto_buy_series(strat)
    decline = [closes[-1] * (1 - 0.002 * k) for k in range(1, 12)]
    df = make_df(closes + decline)
    assert strat.generate_signal(df) == SELL


def test_crypto_momentum_stop_and_target_use_wider_atr_multiples():
    strat = CryptoMomentumStrategy(stop_atr_mult=2.5, target_atr_mult=5.0)
    df = make_df(list(np.linspace(100, 130, 130)))
    entry = 120.0
    stop = strat.get_stop_loss(entry, df, "long")
    target = strat.get_take_profit(entry, df, "long")
    assert stop < entry < target
    assert (target - entry) == pytest.approx(2 * (entry - stop))  # 2:1 R/R by construction


# ------------------------------------------------------ BTC-relative mean reversion

def _btc_alt_pair(idiosyncratic_shock, seed=4, n=60):
    """A BTC series plus an altcoin that tracks it (beta ~1.5) except for a
    shock applied to `idiosyncratic_shock` of the alt's own recent returns
    (unrelated to BTC), producing a residual-return divergence."""
    rng = np.random.RandomState(seed)
    btc_ret = rng.normal(0, 0.005, n)
    btc_closes = list(50_000 * np.cumprod(1 + btc_ret))
    alt_ret = 1.5 * btc_ret + rng.normal(0, 0.001, n)
    idiosyncratic_shock(alt_ret)
    alt_closes = list(100 * np.cumprod(1 + alt_ret))
    return make_df(btc_closes), make_df(alt_closes)


def test_btc_relative_mean_reversion_excluded_from_registry():
    # Needs a second symbol's data the registry's build_strategy() can't supply.
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_strategy("btc_relative_mean_reversion")


def test_btc_relative_mean_reversion_is_long_only():
    btc_df, alt_df = _btc_alt_pair(lambda r: None)
    strat = BtcRelativeMeanReversionStrategy(btc_df)
    assert strat.allow_short is False


def test_btc_relative_mean_reversion_buys_when_residual_is_cheap():
    # Verified empirically: this shock lands the final bar's residual
    # z-score at ~-2.3, inside the entry zone (<=-2.0, above the -3.0 stop).
    def shock(alt_ret):
        alt_ret[-3:] -= 0.009

    btc_df, alt_df = _btc_alt_pair(shock)
    strat = BtcRelativeMeanReversionStrategy(btc_df, beta_period=30, zscore_period=20)
    assert strat.generate_signal(alt_df) == BUY


def test_btc_relative_mean_reversion_sells_when_residual_reverts():
    # Verified empirically: this shock overshoots then snaps back, landing
    # the final bar's z-score at ~+0.4 (reverted to the mean -> take profit).
    def shock(alt_ret):
        alt_ret[-5:-1] -= 0.012

    btc_df, alt_df = _btc_alt_pair(shock)
    strat = BtcRelativeMeanReversionStrategy(btc_df, beta_period=30, zscore_period=20)
    assert strat.generate_signal(alt_df) == SELL


def test_btc_relative_mean_reversion_holds_when_tracking_btc_normally():
    btc_df, alt_df = _btc_alt_pair(lambda r: None)  # no idiosyncratic shock at all
    strat = BtcRelativeMeanReversionStrategy(btc_df, beta_period=30, zscore_period=20)
    assert strat.generate_signal(alt_df) == HOLD


def test_btc_relative_mean_reversion_stop_below_entry():
    btc_df, alt_df = _btc_alt_pair(lambda r: None)
    strat = BtcRelativeMeanReversionStrategy(btc_df)
    entry = 100.0
    stop = strat.get_stop_loss(entry, alt_df, "long")
    target = strat.get_take_profit(entry, alt_df, "long")
    assert stop < entry < target
