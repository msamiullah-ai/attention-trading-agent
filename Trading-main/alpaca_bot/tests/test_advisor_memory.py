"""Tests for the advisor scoring its own past verdicts.

The interesting question is whether a veto can be scored at all. It can: the
position never existed, but the underlying kept trading, so "would this trade
have worked" is answerable from price alone. These pin that logic, because
getting the sign wrong would teach the model the exact opposite of the truth.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.advisor_memory import (  # noqa: E402
    MATERIAL_MOVE_PCT, AdvisorMemory, judge,
)


def old_ts(minutes=120):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# ------------------------------------------------------------- the scoring

def test_veto_on_a_buy_that_would_have_won_is_wrong():
    """We blocked a trade and the price went the trade's way. That cost money."""
    assert judge("veto", "buy", +3.0) is False


def test_veto_on_a_buy_that_would_have_lost_is_right():
    assert judge("veto", "buy", -3.0) is True


def test_confirm_on_a_buy_that_rose_is_right():
    assert judge("confirm", "buy", +3.0) is True


def test_confirm_on_a_buy_that_fell_is_wrong():
    assert judge("confirm", "buy", -3.0) is False


def test_sell_side_inverts_the_sign():
    """A sell wanted the price down; scoring it like a buy would teach the
    model the exact opposite of what happened."""
    assert judge("confirm", "sell", -3.0) is True
    assert judge("confirm", "sell", +3.0) is False
    assert judge("veto", "sell", +3.0) is True


def test_shrink_scores_like_a_confirm():
    """A shrink still let the trade happen, so it is judged on the trade."""
    assert judge("shrink", "buy", +3.0) is True


@pytest.mark.parametrize("move", [0.0, 0.1, -0.4, MATERIAL_MOVE_PCT - 0.01])
def test_an_immaterial_move_is_not_scored(move):
    """The market did not answer the question. Counting it would let noise
    dominate the whole record."""
    assert judge("veto", "buy", move) is None


# ------------------------------------------------------------- the ledger

def test_a_fresh_verdict_is_not_settled_early():
    m = AdvisorMemory(in_memory=True)
    m.record("AAPL", "buy", "veto", "halted", 100.0)
    assert m.settle({"AAPL": 110.0}) == 0        # too young to judge
    assert m.accuracy()[1] == 0


def test_an_old_verdict_settles():
    m = AdvisorMemory(in_memory=True)
    v = m.record("AAPL", "buy", "veto", "halted", 100.0)
    v.ts = old_ts()
    assert m.settle({"AAPL": 110.0}) == 1
    assert v.correct is False                    # blocked a +10% move
    assert v.move_pct == pytest.approx(10.0)


def test_settling_is_idempotent():
    m = AdvisorMemory(in_memory=True)
    v = m.record("AAPL", "buy", "veto", "r", 100.0)
    v.ts = old_ts()
    m.settle({"AAPL": 110.0})
    assert m.settle({"AAPL": 200.0}) == 0        # already judged; not rescored


def test_a_missing_price_leaves_it_unsettled():
    m = AdvisorMemory(in_memory=True)
    v = m.record("AAPL", "buy", "veto", "r", 100.0)
    v.ts = old_ts()
    assert m.settle({"MSFT": 50.0}) == 0
    assert v.correct is None


def test_accuracy_reports_its_sample_size():
    """100% of two and 100% of forty are the same number, different facts."""
    m = AdvisorMemory(in_memory=True)
    for i in range(3):
        v = m.record("AAPL", "buy", "veto", "r", 100.0)
        v.ts = old_ts()
    m.settle({"AAPL": 90.0})
    rate, n = m.accuracy("veto")
    assert n == 3 and rate == 1.0


def test_accuracy_can_filter_by_action():
    m = AdvisorMemory(in_memory=True)
    for action, price in (("veto", 90.0), ("confirm", 90.0)):
        v = m.record("AAPL", "buy", action, "r", 100.0)
        v.ts = old_ts()
    m.settle({"AAPL": 90.0})
    assert m.accuracy("veto")[0] == 1.0          # right to block a fall
    assert m.accuracy("confirm")[0] == 0.0       # wrong to allow it


# -------------------------------------------------------------- the prompt

def test_prompt_block_is_empty_with_no_history():
    assert AdvisorMemory(in_memory=True).prompt_block("AAPL") == ""


def test_prompt_block_shows_past_calls_and_outcomes():
    m = AdvisorMemory(in_memory=True)
    v = m.record("AAPL", "buy", "veto", "halted pending news", 100.0)
    v.ts = old_ts()
    m.settle({"AAPL": 112.0})
    block = m.prompt_block("AAPL")
    assert "AAPL" in block and "VETO" in block
    assert "WRONG" in block                      # it blocked a +12% move
    assert "halted pending news" in block


def test_prompt_block_is_scoped_to_the_symbol():
    """A full history invites the model to pattern-match on its own output
    rather than on the trade in front of it."""
    m = AdvisorMemory(in_memory=True)
    m.record("MSFT", "buy", "veto", "other name", 100.0)
    assert "other name" not in m.prompt_block("AAPL")


# ------------------------------------------------------------ persistence

def test_round_trips_through_disk(tmp_path):
    p = tmp_path / "verdicts.jsonl"
    m = AdvisorMemory(path=p)
    m.record("AAPL", "buy", "veto", "reason here", 100.0)
    assert AdvisorMemory(path=p).recent_for("AAPL")[0].reason == "reason here"


def test_a_torn_line_does_not_poison_the_record(tmp_path):
    p = tmp_path / "verdicts.jsonl"
    m = AdvisorMemory(path=p)
    m.record("AAPL", "buy", "veto", "r", 100.0)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"symbol": "BAD", "act')
    assert len(AdvisorMemory(path=p)._rows) == 1


def test_settled_rows_survive_a_rewrite(tmp_path):
    p = tmp_path / "verdicts.jsonl"
    m = AdvisorMemory(path=p)
    v = m.record("AAPL", "buy", "veto", "r", 100.0)
    v.ts = old_ts()
    m.settle({"AAPL": 90.0})
    assert AdvisorMemory(path=p).accuracy("veto") == (1.0, 1)
