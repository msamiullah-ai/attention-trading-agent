"""Correlation checking: don't let the bot open two positions that are
really the same bet twice. Nothing in trader.py's per-symbol evaluation loop
looks at what else is currently open, so without this a strategy happily
buys AAPL and MSFT in the same cycle -- economically closer to one position
at double size than two independent ones.
"""

from __future__ import annotations

import pandas as pd

# Pairs known to move together regardless of what a short trailing-correlation
# window happens to compute (e.g. a quiet period can mask a real relationship).
# Checked in both directions.
#
# This is a SEED, not the mechanism. Six hand-written pairs cover 9 of a
# 1000-symbol universe and 6 of its 499,500 possible pairings -- the computed
# matrix is what actually does the work, and this only backstops it where a
# quiet window would hide a relationship everyone knows is there. Adding your
# own belongs in config.yaml (`risk.correlated_pairs`) rather than here, so the
# list follows the universe you are actually trading.
HARDCODED_PAIRS: set[frozenset[str]] = {
    frozenset({"AAPL", "MSFT"}),
    frozenset({"GOOGL", "META"}),
    frozenset({"JPM", "BAC"}),
    frozenset({"XOM", "CVX"}),
    frozenset({"AMZN", "GOOGL"}),
    frozenset({"BTC/USD", "ETH/USD"}),
}


def compute_correlation_matrix(bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise correlation of close-to-close returns across symbols, over
    whatever history each DataFrame already contains (caller controls the
    lookback by how much history it fetched). Symbols with too little
    overlapping data to compute a correlation are simply absent from the
    result, not an error.
    """
    returns = {}
    for symbol, df in bars_by_symbol.items():
        if df is None or len(df) < 2:
            continue
        returns[symbol] = df["close"].pct_change().dropna()

    if len(returns) < 2:
        return pd.DataFrame()

    aligned = pd.DataFrame(returns)
    return aligned.corr()


def is_correlated(
    symbol_a: str, symbol_b: str, corr_matrix: pd.DataFrame | None = None,
    threshold: float = 0.75, extra_pairs: set[frozenset[str]] | None = None,
) -> bool:
    """True if the pair is known-correlated, or measures at/above `threshold`.

    `extra_pairs` comes from config so a universe this module has never seen can
    declare its own relationships without an edit here.
    """
    if symbol_a == symbol_b:
        return True
    pair = frozenset({symbol_a, symbol_b})
    if pair in HARDCODED_PAIRS or (extra_pairs and pair in extra_pairs):
        return True
    if corr_matrix is None or corr_matrix.empty:
        return False
    if symbol_a not in corr_matrix.index or symbol_b not in corr_matrix.columns:
        return False
    value = corr_matrix.loc[symbol_a, symbol_b]
    return bool(pd.notna(value) and abs(value) >= threshold)


def find_correlated_open_position(
    symbol: str, open_symbols: set[str], corr_matrix: pd.DataFrame | None = None,
    threshold: float = 0.75, extra_pairs: set[frozenset[str]] | None = None,
) -> str | None:
    """The first already-open symbol correlated with `symbol`, or None."""
    for open_symbol in open_symbols:
        if is_correlated(symbol, open_symbol, corr_matrix, threshold, extra_pairs):
            return open_symbol
    return None
