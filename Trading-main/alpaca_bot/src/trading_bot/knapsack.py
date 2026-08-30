"""0/1 knapsack: pick the subset of items maximizing total value without
exceeding an integer budget. Textbook DP, O(n * budget) time, O(budget)
space for the value table plus an O(n * budget) table for backtracking
which items were taken.

This module is pure and trading-agnostic on purpose -- scripts/deploy_capital.py
builds trading-specific "items" (a symbol/strategy candidate's cost and its
strategy's historical expectancy as value) and calls into this.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnapsackItem:
    id: str
    weight: int  # cost, in whole currency units (e.g. dollars) -- DP requires integers
    value: float


def knapsack_select(items: list[KnapsackItem], budget: int) -> list[KnapsackItem]:
    """Return the subset of `items` maximizing total value with total
    weight <= budget. Each item is taken at most once (0/1, not unbounded).
    An item whose weight exceeds `budget` on its own is simply never chosen,
    not an error.
    """
    if budget <= 0 or not items:
        return []

    n = len(items)
    dp = [0.0] * (budget + 1)
    taken = [[False] * (budget + 1) for _ in range(n)]

    for i, item in enumerate(items):
        if item.weight < 0:
            continue  # a negative weight breaks the 0/1 descending-loop trick; 0 is fine
        # Iterate capacity downward so each item is only ever added once.
        for b in range(budget, item.weight - 1, -1):
            candidate_value = dp[b - item.weight] + item.value
            if candidate_value > dp[b]:
                dp[b] = candidate_value
                taken[i][b] = True

    selected: list[KnapsackItem] = []
    b = budget
    for i in range(n - 1, -1, -1):
        if taken[i][b]:
            selected.append(items[i])
            b -= items[i].weight

    return list(reversed(selected))
