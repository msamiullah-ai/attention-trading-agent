"""Per-strategy reliability learned from closed trades.

THE LOOP THIS CLOSES

`attention.attend` has always accepted a `reliability` multiplier and nothing
ever passed one, so every strategy kept its full regime-fit weight forever no
matter how often it was wrong. News went in, a decision came out, and the
outcome went nowhere. This is the return path.

WHAT IT LEARNS, PRECISELY

Not "what the news meant" -- no model is trained here and nothing about
language is stored. What accumulates is narrower and checkable: for each
strategy, how often the trades it voted for actually made money. That is a hit
rate, and it is the only thing the trade log can honestly support.

CREDIT GOES TO VOTERS, NOT TO THE PRIMARY

Under fusion every trade is recorded against `config.strategy` regardless of
which strategies drove the vote, so the existing trade log cannot attribute an
outcome to an ensemble member. Entry-time votes are carried on the trade and
settled here instead. A strategy that said HOLD gets neither credit nor blame:
it did not cause the trade, and scoring it for one would be noise.

IT CAN ONLY QUIETEN, NEVER AMPLIFY

Reliability is clamped to 1.0 at the top. A strategy on a hot streak does not
get a louder voice than its regime fit already earns it, because a run of wins
is the single easiest thing to mistake for skill in a small sample. A strategy
that keeps losing does get quieter. Same asymmetry as the LLM advisor, for the
same reason: the two error costs are not equal, so the two authorities should
not be either.

SMALL SAMPLES ARE SHRUNK, NOT TRUSTED

Two losing trades is not evidence a strategy is broken. Every hit rate is
pulled toward a coin-flip prior with `PRIOR_WEIGHT` trades of pseudo-evidence,
so a strategy needs a sustained record before its weight moves. Without this
the first two losses of the day would nearly silence a strategy for the rest of
it -- which is not learning, it is thrashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import jsonl
from .logger import get_logger

log = get_logger(__name__)

# A coin flip. Reliability is measured against this, so a strategy hitting 50%
# is treated as neutral rather than as good or bad.
PRIOR = 0.5
# Trades of pseudo-evidence holding a new strategy at the prior. Ten means a
# strategy needs roughly that many real trades before its record moves its
# weight appreciably.
PRIOR_WEIGHT = 10.0
# Floor, matching attention.UNSUITED: a strategy that has been wrong is quieter,
# never silent, because the one time it is right may be the time that matters.
FLOOR = 0.25
# How many recent outcomes count. Old results describe a market that is gone.
DEFAULT_WINDOW = 50


@dataclass
class Outcome:
    """One strategy's share of one closed trade."""

    ts: str
    strategy: str
    symbol: str
    direction: str          # BUY / SELL, as voted
    won: bool
    r_multiple: float

    def to_dict(self) -> dict:
        return {"ts": self.ts, "strategy": self.strategy, "symbol": self.symbol,
                "direction": self.direction, "won": self.won,
                "r_multiple": round(self.r_multiple, 4)}


class StrategyMemory:
    """Append-only record of which strategies were right, and how often.

    Its own file rather than a column on `trades.csv`: one closed trade produces
    several rows here (one per contributing strategy), so the two do not share a
    shape. Keeping them apart also means nothing in this module can corrupt the
    trade log the risk manager depends on.
    """

    def __init__(self, path: Path | None = None, in_memory: bool = False,
                 window: int = DEFAULT_WINDOW):
        self.window = window
        self.path = None if in_memory else (path or jsonl.state_path("strategy_scores.jsonl"))
        self._rows: list[Outcome] = []
        if self.path is not None:
            self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if self.path is None:
            return
        for d in jsonl.read(self.path):
            self._rows.append(Outcome(
                ts=str(d.get("ts", "")), strategy=str(d.get("strategy", "")),
                symbol=str(d.get("symbol", "")), direction=str(d.get("direction", "")),
                won=bool(d.get("won", False)),
                r_multiple=jsonl.coerce(d.get("r_multiple"), float, 0.0),
            ))

    def _append(self, outcome: Outcome) -> None:
        # In memory first. Losing the write degrades the weighting; it must not
        # lose the outcome for the session that is already running.
        self._rows.append(outcome)
        if self.path is not None:
            jsonl.append(self.path, outcome.to_dict())

    # -- recording -----------------------------------------------------------

    def record(self, votes: dict[str, str], direction: str, symbol: str,
               r_multiple: float) -> int:
        """Settle one closed trade across the strategies that voted for it.

        Only voters matching the direction actually taken are scored. Returns
        how many were credited, which is 0 for a trade with no recorded votes
        (a position opened before fusion was on, say) -- that is not an error,
        it just teaches nothing.
        """
        if not votes or not direction:
            return 0
        won = r_multiple > 0
        scored = 0
        now = datetime.now(timezone.utc).isoformat()
        for strategy, vote in votes.items():
            if vote != direction:
                continue        # said HOLD or the opposite; did not cause this
            self._append(Outcome(ts=now, strategy=strategy, symbol=symbol,
                                 direction=direction, won=won, r_multiple=r_multiple))
            scored += 1
        if scored:
            log.info("LEARN %s %s r=%.2f -> credited %d strateg%s",
                     symbol, "WIN" if won else "LOSS", r_multiple, scored,
                     "y" if scored == 1 else "ies")
        return scored

    # -- reading -------------------------------------------------------------

    def hit_rates(self, window: int | None = None) -> dict[str, tuple[float, int]]:
        """Raw hit rate and sample size per strategy, most recent `window` rows.

        The count is returned alongside the rate because a rate without its
        sample size is unreadable -- 100% of two trades and 100% of forty are
        the same number and completely different facts.
        """
        window = window or self.window
        recent = self._rows[-window:] if window else self._rows
        tally: dict[str, list[int]] = {}
        for row in recent:
            slot = tally.setdefault(row.strategy, [0, 0])
            slot[0] += 1 if row.won else 0
            slot[1] += 1
        return {name: (wins / n, n) for name, (wins, n) in tally.items() if n}

    def reliability(self, window: int | None = None) -> dict[str, float]:
        """Per-strategy multiplier for `attention.attend`, in [FLOOR, 1.0].

        Shrunk toward the prior, then measured against it: a shrunk rate at or
        above 50% returns 1.0 (neutral, no penalty), and below it scales down
        proportionally to the floor. Capped at 1.0 on purpose -- see the module
        docstring on why a winning streak buys no extra volume.
        """
        out: dict[str, float] = {}
        for name, (rate, n) in self.hit_rates(window).items():
            shrunk = ((rate * n) + PRIOR * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT)
            out[name] = max(FLOOR, min(1.0, shrunk / PRIOR))
        return out

    def summary(self) -> str:
        rates = self.hit_rates()
        if not rates:
            return "no outcomes recorded yet"
        rel = self.reliability()
        parts = [f"{n}:{r:.0%}({c}) x{rel.get(n, 1.0):.2f}"
                 for n, (r, c) in sorted(rates.items())]
        return " ".join(parts)
