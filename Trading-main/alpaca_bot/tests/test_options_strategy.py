"""The signal-to-option overlay. Pure, so every branch is checkable offline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.options import CALL, PUT, OptionContract  # noqa: E402
from trading_bot.options_strategy import (  # noqa: E402
    NO_ACTION,
    SELL_CALL,
    SELL_PUT,
    plan_option_trade,
    size_cash_secured_puts,
    size_covered_calls,
)


def opt(right, strike, delta, bid=1.00, ask=1.06, oi=500):
    return OptionContract(
        symbol=f"T260904{'P' if right == PUT else 'C'}{int(strike*1000):08d}",
        underlying="T", strike=strike, expiry="2026-09-04", right=right,
        bid=bid, ask=ask, delta=delta, open_interest=oi)


def chain():
    return ([opt(PUT, s, -(0.40 - 0.05 * i)) for i, s in enumerate([98, 96, 94, 92])] +
            [opt(CALL, s, 0.40 - 0.05 * i) for i, s in enumerate([102, 104, 106, 108])])


def plan(**over):
    base = dict(signal="BUY", spot=100.0, chain=chain(), shares_held=0.0,
                cash_available=50_000.0, options_level=1,
                max_collateral=25_000.0, max_contracts=2)
    base.update(over)
    return plan_option_trade(**base)


# --------------------------------------------------------------------- sizing


def test_puts_are_sized_on_collateral_not_premium():
    # The exposure of a short put is strike x 100 of posted cash, while the
    # premium is a few dollars. Sizing off premium would treat a 22,600
    # obligation as a 100 position.
    assert size_cash_secured_puts(226.0, cash_available=50_000,
                                  max_collateral=50_000, max_contracts=10) == 2
    assert size_cash_secured_puts(226.0, cash_available=10_000,
                                  max_collateral=50_000, max_contracts=10) == 0


def test_collateral_cap_binds_independently_of_cash():
    assert size_cash_secured_puts(100.0, cash_available=1_000_000,
                                  max_collateral=25_000, max_contracts=99) == 2


def test_covered_calls_floor_at_whole_lots():
    # Writing against shares you do not have is a naked short call: unbounded
    # loss and level 3. A partial lot must write nothing.
    assert size_covered_calls(250, max_contracts=99) == 2
    assert size_covered_calls(99, max_contracts=99) == 0
    assert size_covered_calls(500, max_contracts=1) == 1


# ------------------------------------------------------------------ the policy


def test_bullish_with_no_shares_sells_a_cash_secured_put():
    p = plan(signal="BUY", shares_held=0)
    assert p.action == SELL_PUT and p.right == PUT and p.side == "sell"
    assert p.qty >= 1 and p.collateral == p.contract.strike * 100 * p.qty


def test_bullish_while_already_long_does_nothing():
    # A covered call here caps exactly the upside the signal expects.
    p = plan(signal="BUY", shares_held=300)
    assert p.action == NO_ACTION and "cap" in p.reason


def test_bearish_on_held_shares_writes_a_covered_call():
    p = plan(signal="SELL", shares_held=300)
    assert p.action == SELL_CALL and p.right == CALL
    assert p.qty == 2  # max_contracts binds before the 3 lots do
    assert p.collateral == 0.0  # secured by shares, not cash


def test_bearish_with_no_shares_has_no_level_1_expression():
    p = plan(signal="SELL", shares_held=0)
    assert p.action == NO_ACTION
    assert "bearish" in p.reason and "level" in p.reason


def test_flat_signal_on_held_shares_is_the_income_overlay():
    p = plan(signal="HOLD", shares_held=200)
    assert p.action == SELL_CALL


def test_flat_signal_with_no_shares_does_nothing():
    assert plan(signal="HOLD", shares_held=0).action == NO_ACTION


# ------------------------------------------------------------- level gating


def test_buying_premium_is_refused_at_level_1():
    # There is no branch that buys at level 1; the policy only ever sells.
    for sig, shares in (("BUY", 0), ("SELL", 300), ("HOLD", 200)):
        p = plan(signal=sig, shares_held=shares, options_level=1)
        assert p.action in (NO_ACTION, SELL_PUT, SELL_CALL)
        assert not p.action.startswith("buy")


def test_unknown_level_refuses_rather_than_assuming():
    p = plan(signal="BUY", options_level=None)
    assert p.action == NO_ACTION and "unknown" in p.reason


# ------------------------------------------------------------------- refusals


def test_no_spot_is_unknown_not_safe():
    p = plan(spot=0.0)
    assert p.action == NO_ACTION and "unknown, not safe" in p.reason


def test_empty_chain_refuses():
    assert plan(chain=[]).action == NO_ACTION


def test_insufficient_cash_names_the_shortfall():
    p = plan(signal="BUY", cash_available=1_000.0)
    assert p.action == NO_ACTION and "collateral" in p.reason


def test_illiquid_chain_refuses_with_the_liquidity_reason():
    bad = [opt(PUT, 96.0, -0.20, bid=0.10, ask=0.50, oi=2)]
    p = plan(signal="BUY", chain=bad)
    assert p.action == NO_ACTION and "liquidity" in p.reason


def test_every_refusal_carries_a_reason():
    # The reason is what appears in the log when nothing traded, so an empty
    # one turns a decision into an unexplained silence.
    for kw in ({"spot": 0.0}, {"chain": []}, {"signal": "SELL", "shares_held": 0},
               {"options_level": None}, {"cash_available": 0.0}):
        assert plan(**kw).reason


def test_actionable_requires_a_contract_and_a_quantity():
    assert plan(signal="BUY").actionable
    assert not plan(signal="HOLD", shares_held=0).actionable
