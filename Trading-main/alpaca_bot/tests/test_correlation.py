"""Correlation checking tests — synthetic price series, no network access."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.correlation import (  # noqa: E402
    compute_correlation_matrix,
    find_correlated_open_position,
    is_correlated,
)


def make_df(closes) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


def test_hardcoded_pair_is_correlated_with_no_matrix():
    assert is_correlated("AAPL", "MSFT") is True
    assert is_correlated("MSFT", "AAPL") is True  # order-independent


def test_hardcoded_pair_crypto():
    assert is_correlated("BTC/USD", "ETH/USD") is True


def test_unrelated_symbols_not_correlated_with_no_matrix():
    assert is_correlated("AAPL", "KO") is False


def test_symbol_is_always_correlated_with_itself():
    assert is_correlated("AAPL", "AAPL") is True


def test_compute_correlation_matrix_detects_lockstep_prices():
    # Two series that move in perfect lockstep -> correlation ~1.0.
    base = list(np.linspace(100, 120, 40))
    bars = {"AAA": make_df(base), "BBB": make_df([p * 2 for p in base])}
    matrix = compute_correlation_matrix(bars)
    assert matrix.loc["AAA", "BBB"] == pytest.approx(1.0, abs=1e-6)


def test_compute_correlation_matrix_detects_inverse_relationship():
    # Build BBB's *returns* as the exact negation of AAA's -- correlating
    # price levels (e.g. two linspace series) doesn't give -1 in return
    # space, since % returns of an ascending vs descending arithmetic
    # sequence aren't mirror images even though the levels are.
    rng = np.random.RandomState(1)
    a_ret = rng.normal(0, 0.01, 40)
    b_ret = -a_ret
    a = list(100 * np.cumprod(1 + a_ret))
    b = list(100 * np.cumprod(1 + b_ret))
    bars = {"AAA": make_df(a), "BBB": make_df(b)}
    matrix = compute_correlation_matrix(bars)
    assert matrix.loc["AAA", "BBB"] == pytest.approx(-1.0, abs=1e-6)


def test_is_correlated_uses_computed_matrix_above_threshold():
    base = list(np.linspace(100, 120, 40))
    bars = {"AAA": make_df(base), "BBB": make_df([p * 2 for p in base])}
    matrix = compute_correlation_matrix(bars)
    assert is_correlated("AAA", "BBB", matrix, threshold=0.75) is True


def test_is_correlated_false_below_threshold():
    rng = np.random.RandomState(0)
    a = list(100 + rng.normal(0, 1, 60).cumsum())
    b = list(100 + rng.normal(0, 1, 60).cumsum())  # independent random walk
    bars = {"AAA": make_df(a), "BBB": make_df(b)}
    matrix = compute_correlation_matrix(bars)
    # Independent random walks: correlation should not reliably exceed 0.75.
    assert abs(matrix.loc["AAA", "BBB"]) < 0.75


def test_compute_correlation_matrix_empty_with_insufficient_symbols():
    assert compute_correlation_matrix({"AAA": make_df([100, 101, 102])}).empty
    assert compute_correlation_matrix({}).empty


def test_compute_correlation_matrix_skips_symbols_with_too_little_data():
    bars = {"AAA": make_df(list(np.linspace(100, 120, 40))), "BBB": make_df([100])}
    matrix = compute_correlation_matrix(bars)
    assert "BBB" not in matrix.columns


def test_find_correlated_open_position_hardcoded():
    result = find_correlated_open_position("AAPL", {"MSFT", "KO"})
    assert result == "MSFT"


def test_find_correlated_open_position_none_when_nothing_correlated():
    result = find_correlated_open_position("AAPL", {"KO", "PEP"})
    assert result is None


def test_find_correlated_open_position_empty_positions():
    assert find_correlated_open_position("AAPL", set()) is None
