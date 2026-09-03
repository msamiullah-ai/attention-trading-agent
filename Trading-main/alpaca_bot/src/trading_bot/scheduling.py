"""Deficit Round Robin for allocating scarce position slots.

THE PROBLEM THIS SOLVES

`run_cycle` walks `config.symbols` in file order and enters a position the
moment a symbol qualifies. With `max_open_positions: 3` and a 1000-symbol
universe, the three slots go to the first three symbols that happen to signal
-- which is an artefact of where they sit in a YAML file, not of how good the
signal was.

Measured on a live run: the slots went to symbols at indices 649, 871 and 892.
Everything past 892 was refused for the rest of the cycle no matter what it
was showing. A 0.85-agreement signal at index 950 loses to a 0.55-agreement
signal at index 100, every single cycle, forever.

THE ALGORITHM

Deficit Round Robin (Shreedhar & Varghese, "Efficient Fair Queuing using
Deficit Round Robin", SIGCOMM 1995). Each queue holds a deficit counter that
grows by a fixed quantum each round; a queue may send when its deficit covers
the cost, and sending spends it. Queues that keep missing out accumulate
credit until they cannot be passed over any more.

Mapped onto trading, one queue per symbol:

    deficit  = credit accrued from cycles this symbol qualified but lost out
    quantum  = credit added per cycle it was passed over
    cost     = the deficit spent when it finally gets a slot

Selection ranks on `signal score + deficit`, so a strong signal wins
immediately, and a repeatedly-overlooked symbol wins eventually. That second
property is the one file order cannot provide at any price.

WHY DRR AND NOT PLAIN ROUND ROBIN

Plain round robin would rotate slots evenly and ignore signal strength
entirely, which trades one arbitrary rule for another. DRR keeps merit
primary and uses the deficit only as a tiebreaker that grows over time -- so
a genuinely better signal still wins, and a symbol only rises on the strength
of having been repeatedly denied, never on nothing at all.

WHAT IT DELIBERATELY DOES NOT DO

It does not decide WHETHER to trade. Every candidate handed to it has already
passed the strategies, the regime weighting, the risk gates and the advisor.
This picks among survivors when there are more of them than there are slots,
and picking is all it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .logger import get_logger

log = get_logger(__name__)

# Credit added per cycle a qualified symbol is passed over. 0.05 against a
# score on 0..1 means roughly twenty consecutive denials before the deficit
# alone rivals a full-strength signal -- slow enough that merit dominates in
# the short run, fast enough that nothing starves across a session.
DEFAULT_QUANTUM = 0.05
# Ceiling on accrued credit. Without it a symbol ignored for a thousand cycles
# would arrive with an unbeatable deficit and hold a slot hostage against far
# better signals -- fairness turning into its own kind of unfairness.
DEFAULT_MAX_DEFICIT = 0.5


@dataclass
class Candidate:
    """One symbol that has passed every check and is competing for a slot."""

    symbol: str
    score: float                 # 0..1, the fused verdict's agreement
    signal: str                  # BUY / SELL
    payload: dict = field(default_factory=dict)   # whatever the caller needs back

    def __repr__(self) -> str:   # keeps log lines readable
        return f"{self.symbol}({self.signal} {self.score:.2f})"


class DeficitRoundRobin:
    """Fair allocation of a fixed number of slots across competing symbols.

    Holds one deficit counter per symbol for the process lifetime. Deliberately
    in-memory and not persisted: the deficit describes contention within a
    trading session, and yesterday's queue is not information about today's.
    """

    def __init__(self, quantum: float = DEFAULT_QUANTUM,
                 max_deficit: float = DEFAULT_MAX_DEFICIT):
        self.quantum = quantum
        self.max_deficit = max_deficit
        self.deficit: dict[str, float] = {}
        # Instrumentation. A scheduler nobody measures is a scheduler nobody
        # can tell is broken: starvation looks exactly like "that symbol never
        # signalled" from the outside.
        self.waits: dict[str, int] = {}        # consecutive cycles passed over
        self.served: dict[str, int] = {}       # slots won, per symbol
        self.wait_on_service: list[int] = []   # wait length at each award
        # Everyone who has ever competed, including symbols never served. The
        # fairness index is meaningless without it: computed over `served`
        # alone, a scheduler that fed exactly one symbol forever would score a
        # perfect 1.0, because the starved symbols simply would not appear.
        self.contenders: set[str] = set()

    def priority(self, candidate: Candidate) -> float:
        """Merit plus accrued credit. Higher wins."""
        return candidate.score + self.deficit.get(candidate.symbol, 0.0)

    def select(self, candidates: list[Candidate], slots: int) -> list[Candidate]:
        """Choose up to `slots` winners and update every deficit.

        Winners spend their credit; losers accrue a quantum. Ties break on
        symbol name rather than on list position, so the ordering of
        config.yaml stops being a hidden input to the decision -- which was the
        entire bug.
        """
        if slots <= 0 or not candidates:
            # Nothing was allocated, so nobody was passed over. Accruing credit
            # here would reward symbols for cycles in which no slot existed.
            return []

        self.contenders.update(c.symbol for c in candidates)
        ranked = sorted(candidates, key=lambda c: (-self.priority(c), c.symbol))
        winners = ranked[:slots]
        won = {c.symbol for c in winners}

        for c in candidates:
            if c.symbol in won:
                self.served[c.symbol] = self.served.get(c.symbol, 0) + 1
                self.wait_on_service.append(self.waits.get(c.symbol, 0))
                self.waits[c.symbol] = 0
            else:
                self.waits[c.symbol] = self.waits.get(c.symbol, 0) + 1

        for c in candidates:
            if c.symbol in won:
                # Spend the credit rather than zeroing it: a symbol that waited
                # a long time and finally traded starts the next contest even,
                # not behind.
                self.deficit[c.symbol] = max(
                    0.0, self.deficit.get(c.symbol, 0.0) - self.quantum)
            else:
                self.deficit[c.symbol] = min(
                    self.max_deficit,
                    self.deficit.get(c.symbol, 0.0) + self.quantum)

        if len(candidates) > slots:
            log.info("SCHEDULER %d candidates for %d slot(s) -> %s (passed over: %s)",
                     len(candidates), slots,
                     ", ".join(repr(c) for c in winners),
                     ", ".join(c.symbol for c in ranked[slots:][:8]))
        return winners

    def metrics(self) -> dict[str, float]:
        """Standard scheduling measures, so fairness is checkable not asserted.

        `avg_wait` / `max_wait` are the textbook waiting-time measures: cycles a
        candidate spent losing before it was finally served. `max_wait` is the
        starvation indicator -- a number that climbs without bound means some
        symbol is never getting through, which is the failure this scheduler
        exists to prevent and the one hardest to see from the outside.

        `fairness` is Jain's index (Jain, Chiu & Hawe, DEC-TR-301, 1984) over
        slots served:

            J = (sum x_i)^2 / (n * sum x_i^2)

        It runs 1/n (one symbol took everything) to 1.0 (perfectly even). It is
        reported alongside the waits rather than instead of them because an even
        split is not automatically the goal here: a universe where one symbol
        genuinely has the best signal every cycle SHOULD score low, and the
        waits are what say whether that is merit or starvation.
        """
        # Zeros included on purpose -- a symbol that competed and never won is
        # the strongest evidence of unfairness there is, and omitting it would
        # invert the metric.
        counts = [self.served.get(sym, 0) for sym in self.contenders]
        n = len(counts)
        fairness = 1.0
        if n:
            total = sum(counts)
            sq = sum(c * c for c in counts)
            fairness = (total * total) / (n * sq) if sq else 1.0
        waits = self.wait_on_service
        return {
            "avg_wait": (sum(waits) / len(waits)) if waits else 0.0,
            "max_wait": float(max(self.waits.values())) if self.waits else 0.0,
            "fairness": fairness,
            "symbols_served": float(len(self.served)),
            "contenders": float(n),
            "slots_awarded": float(sum(counts)),
        }

    def summary(self, top: int = 5) -> str:
        """The symbols currently owed the most, for the dashboard and logs."""
        if not self.deficit:
            return "no deficits accrued"
        owed = sorted(self.deficit.items(), key=lambda kv: -kv[1])[:top]
        return " ".join(f"{s}:{d:.2f}" for s, d in owed if d > 0) or "all even"
