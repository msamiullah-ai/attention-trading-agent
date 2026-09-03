"""Tests for the equity/options capital split.

The property that matters most is the boring one: the two budgets plus the
reserve must never exceed the account. Everything else is a judgement call that
can be retuned; overcommitment is a bug that costs money on the day it happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.allocation import (  # noqa: E402
    DEFAULT_RESERVE, MIN_VIABLE_COLLATERAL, blend, decide,
)
from trading_bot.regime import BEAR, BULL, SIDEWAYS  # noqa: E402


# ------------------------------------------------------- the safety property

@pytest.mark.parametrize("equity", [0, 500, 1_000, 25_000, 100_000, 1_000_000])
@pytest.mark.parametrize("regime", [BULL, BEAR, SIDEWAYS, "", "garbage"])
def test_budgets_never_exceed_the_account(equity, regime):
    """The whole point of having an allocator at all."""
    a = decide(equity, regime=regime, options_level=3, cheapest_collateral=5_000)
    assert a.total <= equity + 1e-6
    assert a.equity_budget >= 0 and a.options_budget >= 0


def test_a_reserve_is_always_held_back():
    """Cash pays assignment, slippage and a stop that gaps. A book with none
    liquidates something to pay for a surprise."""
    a = decide(100_000, regime=BULL, options_level=3)
    assert a.reserve == pytest.approx(100_000 * DEFAULT_RESERVE)
    assert a.equity_budget + a.options_budget <= 100_000 - a.reserve + 1e-6


def test_zero_equity_allocates_nothing():
    a = decide(0.0, regime=BULL, options_level=3)
    assert a.equity_budget == 0 and a.options_budget == 0


def test_negative_equity_is_clamped():
    assert decide(-5_000, options_level=3).total == 0


# ------------------------------------------------------- hard constraints

def test_unknown_options_level_sends_everything_to_equities():
    """Refusing on unknown rather than guessing -- the same rule the options
    overlay already applies to itself."""
    a = decide(100_000, regime=SIDEWAYS, options_level=None)
    assert a.options_budget == 0
    assert "unknown" in a.reason


def test_level_zero_sends_everything_to_equities():
    a = decide(100_000, regime=SIDEWAYS, options_level=0)
    assert a.options_budget == 0


def test_level_one_permits_options():
    a = decide(100_000, regime=SIDEWAYS, options_level=1, cheapest_collateral=5_000)
    assert a.options_budget > 0


def test_a_small_account_gets_no_options_budget():
    """A $92 strike obliges $9,200. On $1,000 no option is reachable at any
    allocation, so reserving for one would starve the side that can trade."""
    a = decide(1_000, regime=SIDEWAYS, options_level=3, cheapest_collateral=9_200)
    assert a.options_budget == 0
    assert a.equity_budget > 0
    assert "unreachable" in a.reason


def test_the_same_account_gets_options_when_contracts_are_cheap():
    """Mirror of the test above: proves affordability is what blocked it."""
    a = decide(10_000, regime=SIDEWAYS, options_level=3, cheapest_collateral=1_500)
    assert a.options_budget > 0


def test_an_options_budget_too_small_for_one_contract_is_refused():
    """Capital reserved to buy nothing is worse than no reservation."""
    a = decide(20_000, regime=BULL, options_level=3, cheapest_collateral=9_000)
    # BULL tilts 75% to equities, leaving ~$4,500 -- under one contract.
    assert a.options_budget == 0
    assert "below the" in a.reason


def test_missing_collateral_info_does_not_disable_options():
    """A slow chain fetch must not silently switch the whole side off."""
    a = decide(100_000, regime=SIDEWAYS, options_level=3, cheapest_collateral=None)
    assert a.options_budget > 0


# ------------------------------------------------------------ the regime tilt

def test_sideways_leans_options():
    """Premium selling is paid by time passing and hurt by direction."""
    a = decide(100_000, regime=SIDEWAYS, options_level=3, cheapest_collateral=5_000)
    assert a.options_budget > a.equity_budget


def test_a_trend_leans_equities():
    for regime in (BULL, BEAR):
        a = decide(100_000, regime=regime, options_level=3, cheapest_collateral=5_000)
        assert a.equity_budget > a.options_budget, regime


def test_an_unknown_regime_splits_evenly():
    """Not a fallback dressed as a decision: with no read there is no basis."""
    a = decide(100_000, regime="", options_level=3, cheapest_collateral=5_000)
    assert a.equity_budget == pytest.approx(a.options_budget)


def test_the_reason_names_the_lean():
    assert "options" in decide(100_000, regime=SIDEWAYS, options_level=3,
                               cheapest_collateral=5_000).reason
    assert "equities" in decide(100_000, regime=BULL, options_level=3,
                                cheapest_collateral=5_000).reason


# ----------------------------------------------------------- regime blending

def test_blend_takes_the_majority():
    assert blend([BULL, BULL, SIDEWAYS]) == BULL


def test_a_tie_resolves_to_sideways():
    """When the universe disagrees about whether a trend exists, betting that
    it does is the more expensive error."""
    assert blend([BULL, BEAR]) == SIDEWAYS


def test_blend_of_nothing_is_sideways():
    assert blend([]) == SIDEWAYS


def test_blend_is_order_independent():
    assert blend([BULL, BEAR, BULL]) == blend([BEAR, BULL, BULL])


# --------------------------------------------------------------- reporting

def test_summary_states_both_budgets_and_the_reason():
    s = decide(100_000, regime=SIDEWAYS, options_level=3,
               cheapest_collateral=5_000).summary()
    assert "equities" in s and "options" in s and "reserve" in s


# ------------------------------------------------------- the dynamic tilt

from trading_bot.allocation import (  # noqa: E402
    TILT_CEILING, TILT_FLOOR, Conditions, _tilt_from,
)


def tilt(regime, **kw):
    return _tilt_from(regime, Conditions(**kw))[0]


def test_no_conditions_falls_back_to_the_anchor():
    """A missing reading contributes nothing rather than a default."""
    assert _tilt_from(BULL, None)[0] == pytest.approx(0.75)


def test_trend_strength_separates_two_bull_markets():
    """A 75% equity allocation on a technicality is how a directional book gets
    chopped up in a market that was never really trending."""
    assert tilt(BULL, adx=40) > tilt(BULL, adx=15)


def test_high_volatility_pushes_toward_options():
    """Rich premium is what a seller is paid for -- the input the static
    version had no way to express."""
    assert tilt(SIDEWAYS, vol_percentile=0.95) < tilt(SIDEWAYS, vol_percentile=0.05)


def test_low_volatility_pushes_toward_equities():
    """Same short-gamma risk for less compensation."""
    assert tilt(SIDEWAYS, vol_percentile=0.05) > tilt(SIDEWAYS, vol_percentile=0.5)


def test_recent_form_moves_toward_the_side_that_is_working():
    better = tilt(SIDEWAYS, equity_hit_rate=0.7, options_hit_rate=0.2)
    worse = tilt(SIDEWAYS, equity_hit_rate=0.2, options_hit_rate=0.7)
    assert better > worse


def test_form_is_ignored_unless_both_sides_have_a_record():
    """One side's hit rate alone says nothing about the comparison."""
    assert tilt(SIDEWAYS, equity_hit_rate=0.9) == tilt(SIDEWAYS)


def test_inputs_accumulate():
    """Each contributes; none is allowed to be the whole answer on its own."""
    one = tilt(SIDEWAYS, vol_percentile=1.0)
    both = tilt(SIDEWAYS, vol_percentile=1.0, adx=0.0)
    assert both < one


def test_neither_side_is_ever_switched_fully_off():
    """Conditions that look extreme are exactly when a reading is most likely
    to be wrong."""
    extreme = tilt(SIDEWAYS, adx=0.0, vol_percentile=1.0,
                   equity_hit_rate=0.0, options_hit_rate=1.0)
    assert TILT_FLOOR <= extreme <= TILT_CEILING
    assert extreme > 0


def test_the_opposite_extreme_is_also_clamped():
    v = tilt(BULL, adx=100.0, vol_percentile=0.0,
             equity_hit_rate=1.0, options_hit_rate=0.0)
    assert v == TILT_CEILING


@pytest.mark.parametrize("adx", [-5.0, 0.0, 20.0, 100.0, 1e9])
@pytest.mark.parametrize("vol", [-1.0, 0.0, 0.5, 1.0, 2.0])
def test_out_of_range_inputs_are_clamped_not_crashed(adx, vol):
    v = tilt(SIDEWAYS, adx=adx, vol_percentile=vol)
    assert TILT_FLOOR <= v <= TILT_CEILING


def test_the_reason_shows_every_contribution():
    """An allocation nobody can audit is one nobody should trust."""
    a = decide(100_000, regime=BULL, options_level=3, cheapest_collateral=5_000,
               conditions=Conditions(adx=40, vol_percentile=0.9))
    assert "anchor" in a.reason and "adx" in a.reason and "vol pct" in a.reason


def test_conditions_still_respect_the_hard_constraints():
    """No reading may unlock options on an account that cannot afford one."""
    a = decide(1_000, regime=SIDEWAYS, options_level=3, cheapest_collateral=9_200,
               conditions=Conditions(vol_percentile=1.0))
    assert a.options_budget == 0


def test_conditions_never_break_the_budget_ceiling():
    for adx in (0.0, 20.0, 60.0):
        for vol in (0.0, 0.5, 1.0):
            a = decide(50_000, regime=SIDEWAYS, options_level=3,
                       cheapest_collateral=5_000,
                       conditions=Conditions(adx=adx, vol_percentile=vol))
            assert a.total <= 50_000 + 1e-6
