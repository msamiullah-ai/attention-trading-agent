"""Knapsack allocator tests — pure DP correctness, no network access."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_bot.knapsack import KnapsackItem, knapsack_select  # noqa: E402


def items(*pairs):
    return [KnapsackItem(id=str(i), weight=w, value=v) for i, (w, v) in enumerate(pairs)]


def test_classic_textbook_example():
    # Well-known small case: optimal is items (2,3)+(3,4) = weight 5, value 7.
    candidates = items((2, 3), (3, 4), (4, 5), (5, 6))
    selected = knapsack_select(candidates, budget=5)
    assert sum(i.value for i in selected) == 7
    assert sum(i.weight for i in selected) <= 5


def test_empty_items_returns_empty():
    assert knapsack_select([], budget=100) == []


def test_zero_budget_returns_empty():
    assert knapsack_select(items((1, 5), (2, 10)), budget=0) == []


def test_single_item_that_fits_is_selected():
    candidates = items((10, 5))
    assert knapsack_select(candidates, budget=10) == candidates


def test_single_item_too_heavy_is_excluded():
    candidates = items((100, 5))
    assert knapsack_select(candidates, budget=10) == []


def test_never_exceeds_budget():
    candidates = items((7, 10), (11, 15), (3, 4), (9, 12), (5, 6), (2, 3))
    selected = knapsack_select(candidates, budget=20)
    assert sum(i.weight for i in selected) <= 20


def test_prefers_higher_value_when_weights_tie():
    candidates = items((10, 5), (10, 50))
    selected = knapsack_select(candidates, budget=10)
    assert len(selected) == 1
    assert selected[0].value == 50


def test_takes_multiple_items_when_it_beats_a_single_expensive_one():
    # Two cheap, valuable items beat one expensive, less-valuable item.
    candidates = items((3, 10), (3, 10), (6, 15))
    selected = knapsack_select(candidates, budget=6)
    assert sum(i.value for i in selected) == 20
    assert len(selected) == 2


def test_zero_weight_positive_value_item_is_always_taken():
    candidates = items((0, 5), (10, 3))
    selected = knapsack_select(candidates, budget=10)
    ids = {i.id for i in selected}
    assert "0" in ids  # the free item


def test_does_not_take_an_item_twice():
    # A single (5, 100) item with budget=20 should select it once, not four times.
    candidates = items((5, 100))
    selected = knapsack_select(candidates, budget=20)
    assert len(selected) == 1


def test_optimal_beats_greedy_by_value_density():
    # Greedy-by-value/weight would take (1,1) six times if unbounded; 0/1
    # knapsack can only take it once. True optimum here is items b+c.
    a = KnapsackItem("a", weight=1, value=1)   # best density but only one exists
    b = KnapsackItem("b", weight=3, value=5)
    c = KnapsackItem("c", weight=3, value=5)
    selected = knapsack_select([a, b, c], budget=6)
    assert sum(i.value for i in selected) == 10
    assert {i.id for i in selected} == {"b", "c"}
