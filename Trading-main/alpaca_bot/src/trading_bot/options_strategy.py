"""Turning a directional signal into an option trade.

Deliberately NOT a `Strategy` subclass. That interface returns BUY/SELL/HOLD
plus `get_stop_loss` and `get_take_profit` -- prices on the underlying, which
an option position does not have. Implementing it here would mean stubbing two
abstract methods with numbers that mean nothing, so instead this consumes the
signal an existing strategy already produces and leaves `signals.py`, the
`REGISTRY` and the current tests alone.

The whole module is pure. Given a signal, a spot, a chain and the account
state, it returns a plan or a refusal with the reason attached. Nothing here
places an order, so every branch is testable without an account.

WHAT THE ACCOUNT LEVEL ALLOWS, because it decides most of this:

    Level 1   sell cash-secured puts, sell covered calls   (collect premium)
    Level 2   the above plus buying calls and puts         (pay premium)
    Level 3   spreads

The asymmetry matters and is not a design choice: at Level 1 there is NO way to
express a bearish view in options. A naked short call is Level 3 and a long put
is Level 2, so a SELL signal on a name we do not own has no option expression
at all. Stated here rather than discovered from a rejection.

THE POLICY, in one table:

    signal  shares held  action
    ------  -----------  -------------------------------------------------
    BUY     no           sell a cash-secured put -- bullish, collects
                         premium, and assignment buys the shares we wanted
    BUY     yes          nothing. A covered call caps the upside we just
                         said we expect; collecting premium against our own
                         thesis is not an overlay, it is a hedge we did not
                         ask for
    SELL    yes          sell a covered call -- premium against a position
                         we are already willing to give up at the strike
    SELL    no           nothing at Level 1 (see the asymmetry above)
    HOLD    yes          sell a covered call -- the income overlay, on shares
                         with no directional view attached
    HOLD    no           nothing

Selling a put on a BUY and a call on a SELL is the wheel: acquire on weakness,
write against what you hold, repeat. What is added on top is that the leg is
chosen by the equity strategy's own signal rather than by rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .options import CALL, PUT, OptionContract, can_trade, select_contract

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"

NO_ACTION = "none"
SELL_PUT = "sell_put"
SELL_CALL = "sell_call"
BUY_CALL = "buy_call"
BUY_PUT = "buy_put"


@dataclass
class OptionPlan:
    """What to do, and why. `reason` is logged verbatim either way."""

    action: str
    reason: str
    contract: OptionContract | None = None
    side: str = ""
    right: str = ""
    qty: int = 0
    collateral: float = 0.0

    @property
    def actionable(self) -> bool:
        return self.action != NO_ACTION and self.contract is not None and self.qty > 0

    def row(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "symbol": self.contract.symbol if self.contract else None,
            "strike": self.contract.strike if self.contract else None,
            "right": self.right,
            "side": self.side,
            "qty": self.qty,
            "collateral": round(self.collateral, 2),
        }


def _nothing(reason: str) -> OptionPlan:
    return OptionPlan(action=NO_ACTION, reason=reason)


def size_cash_secured_puts(strike: float, cash_available: float,
                           max_collateral: float, max_contracts: int) -> int:
    """How many puts the cash actually secures.

    Sized on COLLATERAL, not premium. This is the part an equity risk model
    gets wrong: a short put's exposure is strike x 100 of cash that must be
    posted, while the premium received is a few dollars. Sizing off the premium
    would treat a $22,600 obligation as a $100 position.
    """
    per_contract = strike * 100.0
    if per_contract <= 0:
        return 0
    affordable = int(min(cash_available, max_collateral) // per_contract)
    return max(0, min(affordable, max_contracts))


def size_covered_calls(shares_held: float, max_contracts: int) -> int:
    """One contract per 100 shares. Never more.

    Writing calls against shares you do not have is a naked short call --
    unbounded loss and Level 3 -- so this floors rather than rounds. A partial
    lot writes nothing.
    """
    return max(0, min(int(shares_held // 100), max_contracts))


def plan_option_trade(
    *,
    signal: str,
    spot: float,
    chain: list[OptionContract],
    shares_held: float,
    cash_available: float,
    options_level: int | None,
    max_collateral: float,
    max_contracts: int = 1,
    target_delta: float = 0.20,
    delta_min: float = 0.15,
    delta_max: float = 0.35,
) -> OptionPlan:
    """Translate one directional signal into one option trade, or refuse."""
    signal = (signal or HOLD).upper()

    if spot <= 0:
        return _nothing("no spot price; an unquoted underlying is unknown, not safe")
    if not chain:
        return _nothing("no option chain available for this underlying")

    # --- choose the leg ----------------------------------------------------
    if signal == BUY and shares_held < 100:
        side, right, why = "sell", PUT, "bullish signal, no shares: sell a cash-secured put"
    elif signal == BUY:
        return _nothing("bullish signal but shares already held; a covered call "
                        "would cap exactly the upside the signal expects")
    elif signal == SELL and shares_held >= 100:
        side, right, why = "sell", CALL, "bearish signal on held shares: write a covered call"
    elif signal == SELL:
        return _nothing("bearish signal with no shares; level 1 has no bearish "
                        "option expression (short call is level 3, long put level 2)")
    elif shares_held >= 100:
        side, right, why = "sell", CALL, "no directional view on held shares: income overlay"
    else:
        return _nothing("no signal and no shares; nothing to write against")

    # --- is the account allowed to do it? ----------------------------------
    permitted, level_why = can_trade(side, right, options_level)
    if not permitted:
        return _nothing(level_why)

    # --- pick the contract -------------------------------------------------
    contract, pick_why = select_contract(
        chain, right, spot, target_delta=target_delta,
        delta_min=delta_min, delta_max=delta_max)
    if contract is None:
        return _nothing(f"no tradeable {right}: {pick_why}")

    # --- size it -----------------------------------------------------------
    if right == PUT:
        qty = size_cash_secured_puts(contract.strike, cash_available,
                                     max_collateral, max_contracts)
        if qty == 0:
            need = contract.strike * 100.0
            return _nothing(
                f"cash-secured put needs {need:,.0f} of collateral per contract; "
                f"{min(cash_available, max_collateral):,.0f} available")
        collateral = contract.collateral(qty)
    else:
        qty = size_covered_calls(shares_held, max_contracts)
        if qty == 0:
            return _nothing(f"covered call needs 100 shares per contract; "
                            f"holding {shares_held:g}")
        collateral = 0.0  # secured by the shares, not by cash

    return OptionPlan(
        action=SELL_PUT if right == PUT else SELL_CALL,
        reason=f"{why}; {pick_why}; {level_why}",
        contract=contract, side=side, right=right, qty=qty,
        collateral=collateral)
