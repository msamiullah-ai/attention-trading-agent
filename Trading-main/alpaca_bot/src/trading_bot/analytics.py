"""Behavioral analytics on the trade journal: the questions most traders
never actually check systematically -- are you cutting winners short before
they reach the planned target, or letting losers run past the planned stop?
Computed from the same TradeLog risk.py already maintains; nothing here
needs new data collection.
"""

from __future__ import annotations

from dataclasses import dataclass

from .risk import TradeRecord


@dataclass
class BehavioralStats:
    n: int
    planned_win_r_avg: float | None  # avg planned reward/risk on trades that ended up winners
    actual_win_r_avg: float | None   # avg realized R on those same winners
    actual_loss_r_avg: float | None  # avg realized |R| on losers (planned is always 1R, by definition)
    cutting_winners_short: bool
    letting_losers_run: bool

    def summary_lines(self) -> list[str]:
        lines = []
        if self.planned_win_r_avg is not None and self.actual_win_r_avg is not None:
            lines.append(f"Planned win R/R: {self.planned_win_r_avg:.2f}  Actual: {self.actual_win_r_avg:.2f}")
            if self.cutting_winners_short:
                lines.append("  WARNING: cutting winners short of their planned target.")
        if self.actual_loss_r_avg is not None:
            lines.append(f"Actual loss R (planned is always 1.0R): {self.actual_loss_r_avg:.2f}")
            if self.letting_losers_run:
                lines.append("  WARNING: losses are running past the planned 1R stop.")
        if not lines:
            lines.append("Not enough trade history with a recorded target yet.")
        return lines


def behavioral_stats(
    trades: list[TradeRecord],
    early_exit_threshold: float = 0.70,
    late_stop_threshold: float = 1.30,
) -> BehavioralStats:
    """`early_exit_threshold`: flag if avg actual win R falls below this
    fraction of the avg planned win R (default: winners realizing less than
    70% of their planned target, on average).
    `late_stop_threshold`: flag if avg actual loss magnitude exceeds this
    multiple of 1R (default: losses running 30%+ past the planned stop).
    """
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    wins_with_plan = [t for t in wins if t.planned_reward_risk is not None]
    planned_win_avg = (
        sum(t.planned_reward_risk for t in wins_with_plan) / len(wins_with_plan)
        if wins_with_plan else None
    )
    # Actual R is measured only over the same subset that has a recorded
    # plan, so "planned vs actual" compares like with like.
    actual_win_avg = (
        sum(t.r_multiple for t in wins_with_plan) / len(wins_with_plan)
        if wins_with_plan else None
    )
    actual_loss_avg = sum(abs(t.r_multiple) for t in losses) / len(losses) if losses else None

    cutting_winners_short = bool(
        planned_win_avg is not None and actual_win_avg is not None
        and planned_win_avg > 0 and actual_win_avg < early_exit_threshold * planned_win_avg
    )
    letting_losers_run = bool(
        actual_loss_avg is not None and actual_loss_avg > late_stop_threshold
    )

    return BehavioralStats(
        n=len(trades),
        planned_win_r_avg=planned_win_avg,
        actual_win_r_avg=actual_win_avg,
        actual_loss_r_avg=actual_loss_avg,
        cutting_winners_short=cutting_winners_short,
        letting_losers_run=letting_losers_run,
    )
