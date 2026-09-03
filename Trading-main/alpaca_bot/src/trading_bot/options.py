"""Option contracts: parsing, selection, and collateral.

Everything here is pure except `fetch_chain`, so strike selection and the risk
arithmetic are testable without a network or an account -- the same property
`indicators.py` and `signals.py` already have in this codebase.

Covers calls and puts, both directions. What an account may actually trade is
decided by its options level, not by this module:

    Level 1   covered calls, cash-secured puts        (selling premium)
    Level 2   the above plus buying calls and puts    (long premium)
    Level 3   spreads and multi-leg

So `select_contract` will happily find you a long call; whether Alpaca fills it
depends on approval this code cannot see. `LEVEL_REQUIRED` records which is
which so a strategy can refuse rather than discover it from a rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

CALL = "call"
PUT = "put"

# What each intent needs from the account. Checked before ordering so an
# unapproved strategy fails with a clear message instead of a broker rejection.
LEVEL_REQUIRED = {
    ("sell", PUT): 1,    # cash-secured put
    ("sell", CALL): 1,   # covered call -- only if the shares are held
    ("buy", CALL): 2,
    ("buy", PUT): 2,
}


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    strike: float
    expiry: str          # ISO yyyy-mm-dd
    right: str           # "call" | "put"
    bid: float | None = None
    ask: float | None = None
    delta: float | None = None
    implied_vol: float | None = None
    open_interest: int | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float | None:
        """Relative spread. The liquidity number that decides whether a fill is
        possible at a sane price -- an option can look cheap and be untradeable."""
        m = self.mid
        if m is None or m <= 0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / m

    def collateral(self, qty: int = 1) -> float:
        """Cash a SHORT position must have posted behind it.

        A short put is cash-secured at strike x 100: if assigned you buy the
        shares at the strike, and that is the trade being sold, not an accident.
        A short call is share-secured -- 100 shares per contract, no cash -- so
        it returns 0.0 here and the share check is the caller's job.
        """
        return self.strike * 100.0 * qty if self.right == PUT else 0.0

    def dte(self, today: date) -> int:
        return days_to_expiry(self.expiry, today)


def parse_occ(symbol: str) -> OptionContract | None:
    """Parse an OCC symbol: ROOT + YYMMDD + C/P + strike x 1000, 8 digits.

    Strict on purpose, returning None rather than a guess. The lax version --
    `int(symbol[-8:]) / 1000` -- yields 0.0 for anything unexpected, which
    silently zeroes the collateral of a position and understates deployment.
    A risk figure that fails quietly is worse than one that fails loudly.
    """
    s = (symbol or "").strip().upper()
    if len(s) < 16:                      # 1 root + 6 date + 1 right + 8 strike
        return None
    strike_part, right_ch, date_part, root = s[-8:], s[-9], s[-15:-9], s[:-15]
    if not (strike_part.isdigit() and date_part.isdigit() and right_ch in "CP"):
        return None
    if not root or not root.isalpha():
        return None
    try:
        yy, mm, dd = int(date_part[:2]), int(date_part[2:4]), int(date_part[4:6])
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            return None
        strike = int(strike_part) / 1000.0
    except ValueError:
        return None
    if strike <= 0:
        return None
    return OptionContract(
        symbol=s, underlying=root, strike=strike,
        expiry=f"20{yy:02d}-{mm:02d}-{dd:02d}",
        right=CALL if right_ch == "C" else PUT,
    )


def days_to_expiry(expiry: str, today: date) -> int:
    """Calendar days, floored at zero.

    `today` must be the MARKET date. A machine in another timezone can be a day
    ahead, which makes a contract read one day shorter than it is -- and DTE
    feeds both premium-per-day and the assignment window.
    """
    try:
        y, m, d = (int(x) for x in expiry.split("-"))
    except (ValueError, AttributeError):
        return 0
    return max((date(y, m, d) - today).days, 0)


def moneyness(contract: OptionContract, spot: float) -> float:
    """How far out of the money, as a fraction of spot. Negative means ITM.

    Signed the same way for both rights, so a caller does not have to remember
    which direction hurts: positive is always the safe side of the strike.
    """
    if spot <= 0:
        return 0.0
    if contract.right == PUT:
        return (spot - contract.strike) / spot
    return (contract.strike - spot) / spot


def is_itm(contract: OptionContract, spot: float) -> bool:
    return moneyness(contract, spot) <= 0


def select_contract(
    chain: list[OptionContract],
    right: str,
    spot: float,
    *,
    target_delta: float = 0.20,
    delta_min: float = 0.15,
    delta_max: float = 0.35,
    otm_min: float = 0.01,
    otm_max: float = 0.06,
    max_spread_pct: float = 0.20,
    min_open_interest: int = 50,
) -> tuple[OptionContract | None, str]:
    """Pick one contract. Returns (contract, why) -- `why` explains either the
    choice or the refusal, and is meant to be logged verbatim.

    Delta is primary because it is the market's own estimate of the probability
    of finishing in the money, which is the risk actually being taken.

    Percentage moneyness is the fallback when delta is missing, which happens
    often on thin chains. Black-Scholes is deliberately NOT the fallback: it
    needs an implied vol the feed also failed to provide, so it would fabricate
    the number being claimed as missing.

    The fallback must not become a way around the delta band. A contract whose
    delta is KNOWN and outside the band is rejected, not rescued by moneyness --
    otherwise the band stops being a limit for exactly the contracts that most
    need one.
    """
    right = right.lower()
    live = [c for c in chain
            if c.right == right and c.bid is not None and c.bid > 0]
    if not live:
        return None, f"no {right} in the chain has a live bid"

    def liquid(c: OptionContract) -> bool:
        sp = c.spread_pct
        if sp is not None and sp > max_spread_pct:
            return False
        if c.open_interest is not None and c.open_interest < min_open_interest:
            return False
        return True

    tradeable = [c for c in live if liquid(c)]
    if not tradeable:
        return None, (f"all {len(live)} {right}s failed liquidity "
                      f"(spread > {max_spread_pct:.0%} or OI < {min_open_interest})")

    banded = [c for c in tradeable
              if c.delta is not None and delta_min <= abs(c.delta) <= delta_max]
    if banded:
        best = min(banded, key=lambda c: abs(abs(c.delta) - target_delta))
        return best, f"delta {best.delta:+.3f} (target {target_delta})"

    if spot > 0:
        window = [c for c in tradeable if otm_min <= moneyness(c, spot) <= otm_max]
        window = [c for c in window
                  if c.delta is None or delta_min <= abs(c.delta) <= delta_max]
        if window:
            target = (otm_min + otm_max) / 2
            best = min(window, key=lambda c: abs(moneyness(c, spot) - target))
            return best, (f"moneyness {moneyness(best, spot):.2%} OTM "
                          f"(delta unavailable)")

    return None, "no contract inside the delta band or the moneyness window"


def can_trade(side: str, right: str, options_level: int | None) -> tuple[bool, str]:
    """Whether this account level permits this intent.

    Checked before ordering so the refusal names the reason. Discovering it
    from a broker rejection instead means the log says "order failed" when it
    means "this account was never allowed to do that".
    """
    need = LEVEL_REQUIRED.get((side.lower(), right.lower()))
    if need is None:
        return False, f"unknown intent {side} {right}"
    if options_level is None:
        return False, "account options level unknown"
    if options_level < need:
        return False, (f"{side} {right} needs options level {need}, "
                       f"account has {options_level}")
    return True, f"level {options_level} permits {side} {right}"
