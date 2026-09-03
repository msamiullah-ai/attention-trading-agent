"""The advisor's record of its own past verdicts, and what they cost.

THE PROBLEM

`agent.py` is stateless. It can veto a symbol today, be wrong, and veto it
identically tomorrow, forever, with nothing anywhere noticing. Every other
layer in this bot learns something; the one making judgement calls in prose
learns nothing at all.

THE COUNTERFACTUAL, WHICH IS NOT ACTUALLY MISSING

The usual objection is that a veto has no measurable outcome: the trade was
never placed, so there is no P&L to score. That is true of the *position* and
false of the *decision*. The price keeps trading whether or not we participated.

    vetoed a BUY, price then rose   -> the veto cost us
    vetoed a BUY, price then fell   -> the veto saved us
    confirmed a BUY, price fell     -> the confirm cost us

So each verdict is settled against the underlying's own subsequent move over a
fixed horizon. That is a real, checkable score for advice -- not a proxy, and
not the same thing as trade P&L, which is why it lives here rather than in the
trade log.

WHAT IT DOES WITH THE RECORD

Two things, and deliberately not a third.

  1. It measures. `accuracy()` reports how often the advisor's vetoes were
     right, so "is this model worth its latency" stops being a matter of
     opinion.

  2. It shows the model its own recent calls, so it can notice when it has
     vetoed the same name four times running on the same reasoning.

What it does NOT do is auto-adjust the advisor's authority based on its score.
A veto is already the safe direction, the sample is small, and quietly widening
a model's power because it had a good week is exactly the kind of feedback loop
that is invisible until it is expensive. The number is reported to a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import jsonl
from .logger import get_logger

log = get_logger(__name__)

# How far ahead a verdict is judged. Long enough that a single bar's noise does
# not decide it, short enough to stay relevant to an intraday decision.
SETTLE_HORIZON_MIN = 60
# Move required before a verdict counts as right or wrong either way. Below
# this the market did not answer the question, and scoring it would be scoring
# noise.
MATERIAL_MOVE_PCT = 0.5


@dataclass
class Verdict:
    """One piece of advice, and how it turned out."""

    ts: str
    symbol: str
    side: str                     # the trade being reviewed: buy / sell
    action: str                   # confirm / shrink / veto
    reason: str
    price_at: float               # underlying price when advised
    settled_price: float | None = None
    move_pct: float | None = None
    correct: bool | None = None   # None = not yet settled, or immaterial move

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "symbol": self.symbol, "side": self.side,
            "action": self.action, "reason": self.reason[:200],
            "price_at": round(self.price_at, 4),
            "settled_price": None if self.settled_price is None else round(self.settled_price, 4),
            "move_pct": None if self.move_pct is None else round(self.move_pct, 3),
            "correct": self.correct,
        }

    @staticmethod
    def from_dict(d: dict) -> "Verdict":
        return Verdict(
            ts=str(d.get("ts", "")), symbol=str(d.get("symbol", "")),
            side=str(d.get("side", "")), action=str(d.get("action", "")),
            reason=str(d.get("reason", "")), price_at=float(d.get("price_at") or 0.0),
            settled_price=(None if d.get("settled_price") is None
                           else float(d["settled_price"])),
            move_pct=None if d.get("move_pct") is None else float(d["move_pct"]),
            correct=d.get("correct"),
        )


def judge(action: str, side: str, move_pct: float) -> bool | None:
    """Was this verdict right, given what the underlying then did?

    Returns None when the move was too small to call. That is not a hedge: a
    0.05% drift is not evidence about anything, and counting it would let noise
    dominate the record entirely.
    """
    if abs(move_pct) < MATERIAL_MOVE_PCT:
        return None
    # Did the trade we were reviewing want the price to go up or down?
    wanted_up = side.lower() == "buy"
    price_helped = (move_pct > 0) if wanted_up else (move_pct < 0)
    if action == "veto":
        return not price_helped        # right to block only if it would have lost
    return price_helped                # confirm/shrink: right if the trade worked


class AdvisorMemory:
    """Append-only log of advice, settled against later prices."""

    def __init__(self, path: Path | None = None, in_memory: bool = False,
                 window: int = 50):
        self.window = window
        self.path = None if in_memory else (path or jsonl.state_path("advisor_verdicts.jsonl"))
        self._rows: list[Verdict] = []
        if self.path is not None:
            self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        for d in jsonl.read(self.path):
            try:
                self._rows.append(Verdict.from_dict(d))
            except (ValueError, TypeError):
                continue          # a row from an older schema

    def _rewrite(self) -> None:
        """Rewrite the whole file. Rows are mutated in place when they settle,
        so append-only does not work for updates -- and the file is bounded by
        the number of trades reviewed, which is small."""
        if self.path is None:
            return
        jsonl.rewrite(self.path, (v.to_dict() for v in self._rows), keep=500)

    # -- recording -----------------------------------------------------------

    def record(self, symbol: str, side: str, action: str, reason: str,
               price: float) -> Verdict:
        v = Verdict(ts=datetime.now(timezone.utc).isoformat(), symbol=symbol,
                    side=side, action=action, reason=reason, price_at=price)
        self._rows.append(v)
        self._rewrite()
        return v

    def settle(self, prices: dict[str, float]) -> int:
        """Score every unsettled verdict for which a current price is known.

        Called once per cycle with the prices already fetched for trading, so
        settling costs no extra API calls.
        """
        settled = 0
        for row in self._rows:
            if row.correct is not None or row.settled_price is not None:
                continue
            price_now = prices.get(row.symbol)
            if price_now is None or row.price_at <= 0:
                continue
            if not self._old_enough(row.ts):
                continue
            row.settled_price = price_now
            row.move_pct = (price_now - row.price_at) / row.price_at * 100.0
            row.correct = judge(row.action, row.side, row.move_pct)
            settled += 1
        if settled:
            self._rewrite()
            log.info("ADVISOR SCORED %d verdict(s) | %s", settled, self.summary())
        return settled

    @staticmethod
    def _old_enough(ts: str) -> bool:
        try:
            then = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return False
        age_min = (datetime.now(timezone.utc) - then).total_seconds() / 60.0
        return age_min >= SETTLE_HORIZON_MIN

    # -- reading -------------------------------------------------------------

    def accuracy(self, action: str | None = None) -> tuple[float, int]:
        """Hit rate and sample size over settled verdicts.

        The count comes back with the rate because one without the other is
        unreadable -- 100% of two calls and 100% of forty are the same number
        and completely different facts.
        """
        rows = [r for r in self._rows[-self.window:]
                if r.correct is not None and (action is None or r.action == action)]
        if not rows:
            return (0.0, 0)
        return (sum(1 for r in rows if r.correct) / len(rows), len(rows))

    def recent_for(self, symbol: str, limit: int = 3) -> list[Verdict]:
        return [r for r in self._rows if r.symbol == symbol][-limit:]

    def prompt_block(self, symbol: str) -> str:
        """What the model is shown about its own past calls.

        Scoped to this symbol plus one overall accuracy line. A full history
        would be both enormous and an invitation to pattern-match on its own
        output rather than on the trade in front of it -- the useful signal is
        narrow: have I said this before about this name, and was I right.
        """
        lines: list[str] = []
        rate, n = self.accuracy("veto")
        if n:
            lines.append(f"YOUR VETO ACCURACY: {rate:.0%} over {n} settled call(s)")
        past = self.recent_for(symbol)
        if past:
            lines.append(f"YOUR RECENT CALLS ON {symbol}:")
            for v in past:
                if v.correct is None:
                    outcome = "not yet settled"
                else:
                    outcome = (f"{'RIGHT' if v.correct else 'WRONG'}, "
                               f"underlying moved {v.move_pct:+.2f}%")
                lines.append(f"  - {v.ts[:16]} {v.action.upper()}: "
                             f"{v.reason[:90]} -> {outcome}")
        return "\n".join(lines)

    def summary(self) -> str:
        overall, n = self.accuracy()
        veto, vn = self.accuracy("veto")
        if not n:
            return "no verdicts settled yet"
        return f"overall {overall:.0%} ({n})  veto {veto:.0%} ({vn})"
