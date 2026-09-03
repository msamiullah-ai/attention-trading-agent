"""Deciding how much capital goes to equities and how much to options.

THE PROBLEM

`main.py` and `main_options.py` are separate processes. Each reads the account
and sizes against the FULL equity: the equity side may commit 3 positions at 5%
each, the options side may post 50% of equity as collateral, and neither knows
the other exists. Run both and the account can be committed past 100% with no
component doing anything wrong by its own rules.

There is also no notion anywhere of *preferring* one. "The market is good for
premium selling right now, lean options" is not a decision this bot can express.

WHAT THIS DOES

Splits the account into two budgets that sum to at most the tradeable equity,
and hands one to each side. Overcommitment stops being a thing that requires
coordination and becomes a thing the arithmetic forbids.

HOW IT DECIDES, AND WHY THAT WAY

Hard constraints first, because they are facts rather than judgements:

  level       Below options level 1 there is no options expression at all, so
              the whole account goes to equities. Level unknown is treated the
              same -- refusing on unknown rather than guessing is the same rule
              the options overlay already applies to itself.

  affordable  A cash-secured put on a $92 strike obliges $9,200. On a $1,000
              account no option is reachable at any allocation, so allocating
              to it would only starve the side that CAN trade. This is checked
              in dollars, not as a percentage, because that is how the
              obligation actually arrives.

Then the regime tilt, which is a judgement and is stated as one:

  SIDEWAYS    favours options. Premium selling is short-vol and short-gamma:
              it is paid by time passing and hurt by direction, so a market
              with no trend to ride is where it earns and a directional
              strategy is where it does not.

  BULL/BEAR   favours equities. A trend is exactly what a directional strategy
              is for, and exactly what a short-premium book has to survive
              rather than profit from.

  unknown     even split. Not a fallback dressed as a decision -- with no read
              on the regime there is no basis to prefer either.

WHAT IT DELIBERATELY DOES NOT DO

It does not pick trades, symbols or sizes. It sets two ceilings. Each side then
applies its own risk rules inside its own ceiling, unchanged -- which is why
turning this on cannot make either side take a trade it would otherwise refuse.
It can only ever give them less than they have today, never more.
"""

from __future__ import annotations

from dataclasses import dataclass

from .logger import get_logger
from .regime import BEAR, BULL, SIDEWAYS

log = get_logger(__name__)

# Regime tilts, as the equity share of the tradeable budget. These are the
# ANCHORS, not the answer -- `_tilt_from` moves away from them continuously as
# conditions warrant, so a barely-trending market does not get the same
# allocation as a strongly-trending one.
TILT: dict[str, float] = {
    SIDEWAYS: 0.35,        # lean options: theta is paid, direction is not
    BULL: 0.75,            # lean equities: there is a trend to ride
    BEAR: 0.75,            # same, in the other direction
}
NEUTRAL_TILT = 0.5         # no read on the regime, no basis to prefer either

# How far each live input may move the tilt from its regime anchor. Bounded so
# no single reading can take the whole account: three inputs each worth up to
# 0.15 can shift 0.45, which is decisive without ever being unilateral.
MAX_SHIFT_PER_INPUT = 0.15
# Tilt is clamped here regardless. Neither side is ever switched fully off by
# conditions -- a book with no position in one instrument cannot learn anything
# about it, and conditions that look extreme are exactly when a reading is most
# likely to be wrong.
TILT_FLOOR, TILT_CEILING = 0.15, 0.85

# ADX below this reads as no trend, above as a strong one. The midpoint of the
# regime detector's own sideways/trending thresholds (20/25), so this module
# and `detect_regime` cannot disagree about what "trending" means.
ADX_FLAT, ADX_STRONG = 20.0, 35.0

# Fraction of equity held back from both sides. Not a rounding buffer: it is
# what pays assignment, slippage and a stop that gaps, and a book with no cash
# is a book that liquidates something to pay for a surprise.
DEFAULT_RESERVE = 0.10

# Cheapest option worth the allocator's attention, as collateral. Below roughly
# this, an option's underlying is a penny stock and the contract is illiquid.
MIN_VIABLE_COLLATERAL = 500.0


@dataclass
class Conditions:
    """Live readings that move the allocation away from its regime anchor.

    Every field is optional. A missing reading contributes nothing rather than
    a default, because "I could not measure this" and "this measured neutral"
    are different facts and only one of them should move money.
    """

    adx: float | None = None
    # Current realized vol against its own recent history, 0..1. High means
    # premium is rich relative to what this market normally pays.
    vol_percentile: float | None = None
    # Recent hit rate of each side, 0..1, from the trade log. None until there
    # is enough history to mean anything.
    equity_hit_rate: float | None = None
    options_hit_rate: float | None = None


def _tilt_from(regime: str, c: Conditions | None) -> tuple[float, list[str]]:
    """Continuous equity share, and a plain-language account of how it got there.

    Starts at the regime anchor and moves. Each input is scaled to a signed
    contribution in [-MAX_SHIFT_PER_INPUT, +MAX_SHIFT_PER_INPUT], where positive
    means "toward equities":

      trend strength   A strong ADX means the trend the regime label claims is
                       actually there. A weak one means the label is a
                       technicality, and a 75% equity allocation on a
                       technicality is how a directional book gets chopped up
                       in a market that was never really trending.

      volatility       High realized vol makes premium rich, which is what a
                       seller is paid for -- so it pushes toward options. Low
                       vol means the same short-gamma risk for less
                       compensation, which pushes the other way. This is the
                       input the static version had no way to express at all.

      recent form      Which side has actually been working lately. Bounded
                       hardest in spirit: it is the most tempting input and the
                       most likely to be noise, so it moves the tilt only when
                       both sides have a record to compare.
    """
    tilt = TILT.get(regime, NEUTRAL_TILT)
    why: list[str] = [f"regime {regime or 'unknown'} anchor {tilt:.2f}"]
    if c is None:
        return tilt, why

    if c.adx is not None:
        # -1 at flat, +1 at strongly trending.
        span = max(1e-9, ADX_STRONG - ADX_FLAT)
        strength = max(-1.0, min(1.0, (float(c.adx) - ADX_FLAT) / span * 2.0 - 1.0))
        shift = strength * MAX_SHIFT_PER_INPUT
        tilt += shift
        why.append(f"adx {c.adx:.0f} {shift:+.2f}")

    if c.vol_percentile is not None:
        # High vol -> richer premium -> toward options, hence the negation.
        v = max(0.0, min(1.0, float(c.vol_percentile)))
        shift = -((v - 0.5) * 2.0) * MAX_SHIFT_PER_INPUT
        tilt += shift
        why.append(f"vol pct {v:.2f} {shift:+.2f}")

    if c.equity_hit_rate is not None and c.options_hit_rate is not None:
        edge = max(-1.0, min(1.0, float(c.equity_hit_rate) - float(c.options_hit_rate)))
        shift = edge * MAX_SHIFT_PER_INPUT
        tilt += shift
        why.append(f"form {c.equity_hit_rate:.0%}v{c.options_hit_rate:.0%} {shift:+.2f}")

    clamped = max(TILT_FLOOR, min(TILT_CEILING, tilt))
    if clamped != tilt:
        why.append(f"clamped to {clamped:.2f}")
    return clamped, why


@dataclass
class Allocation:
    """Two ceilings and the reasoning that produced them."""

    equity_budget: float
    options_budget: float
    reserve: float
    regime: str
    reason: str

    @property
    def total(self) -> float:
        return self.equity_budget + self.options_budget + self.reserve

    def summary(self) -> str:
        return (f"equities ${self.equity_budget:,.0f} | options "
                f"${self.options_budget:,.0f} | reserve ${self.reserve:,.0f} "
                f"| {self.reason}")


def decide(
    equity: float,
    regime: str = "",
    options_level: int | None = None,
    cheapest_collateral: float | None = None,
    reserve_pct: float = DEFAULT_RESERVE,
    conditions: "Conditions | None" = None,
) -> Allocation:
    """Split `equity` between the two sides.

    `cheapest_collateral` is what the cheapest option the options side would
    actually consider would tie up (strike x 100). Passing None means "not
    known this cycle", which is treated as affordable rather than as blocking:
    the options overlay does its own affordability check per contract anyway,
    and refusing here on missing information would silently disable the whole
    side because a chain fetch was slow.
    """
    equity = max(0.0, float(equity))
    reserve = equity * max(0.0, min(0.5, reserve_pct))
    tradeable = max(0.0, equity - reserve)

    if tradeable <= 0:
        return Allocation(0.0, 0.0, reserve, regime or "unknown",
                          "no tradeable equity")

    # -- hard constraints ----------------------------------------------------

    if options_level is None or options_level < 1:
        why = ("options level unknown" if options_level is None
               else f"options level {options_level}")
        return Allocation(tradeable, 0.0, reserve, regime or "unknown",
                          f"{why}: no options expression available")

    floor = (cheapest_collateral if cheapest_collateral is not None
             else MIN_VIABLE_COLLATERAL)
    if floor > tradeable:
        return Allocation(
            tradeable, 0.0, reserve, regime or "unknown",
            f"cheapest option needs ${floor:,.0f} collateral, "
            f"${tradeable:,.0f} tradeable: options unreachable")

    # -- regime tilt ---------------------------------------------------------

    tilt, why = _tilt_from(regime, conditions)
    equity_budget = tradeable * tilt
    options_budget = tradeable - equity_budget

    # An options budget that cannot fund one contract is worse than none: it
    # reserves capital the equity side could have used, to buy nothing.
    if options_budget < floor:
        return Allocation(
            tradeable, 0.0, reserve, regime or "unknown",
            f"options share ${options_budget:,.0f} below the ${floor:,.0f} "
            f"needed for one contract: all to equities")

    lean = ("options" if tilt < 0.5 else "equities" if tilt > 0.5 else "neither")
    return Allocation(
        equity_budget, options_budget, reserve, regime or "unknown",
        f"tilt {tilt:.2f} favours {lean} ({'; '.join(why)})")


def blend(regimes: list[str]) -> str:
    """One regime for the account from many per-symbol reads.

    The allocator is an account-level decision but `detect_regime` is per
    symbol, so they have to be reconciled somewhere. Majority wins, and a tie
    resolves to SIDEWAYS -- when the universe disagrees about whether there is
    a trend, betting that there is one is the more expensive error.
    """
    if not regimes:
        return SIDEWAYS
    counts = {r: regimes.count(r) for r in set(regimes)}
    top = max(counts.values())
    winners = sorted(r for r, n in counts.items() if n == top)
    return winners[0] if len(winners) == 1 else SIDEWAYS
