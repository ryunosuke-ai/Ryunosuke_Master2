from __future__ import annotations

from collections import Counter

import pytest

from scripts.prepare_esconv_google_form_likert_blocks import (
    split_category_pairs,
    validate_split,
)


def candidate(index: int, category: str, advantage: float) -> dict:
    return {
        "prompt_id": f"p{index:03d}",
        "category": category,
        "basis_advantage_over_best_control": advantage,
    }


def candidates() -> list[dict]:
    rows = []
    for category_index in range(10):
        category = f"category_{category_index}"
        rows.append(candidate(category_index * 2, category, 0.5 + category_index))
        rows.append(
            candidate(category_index * 2 + 1, category, 1.0 + category_index)
        )
    return rows


def test_split_has_one_item_from_every_category_and_no_overlap():
    rows = candidates()
    experiments = split_category_pairs(rows)
    validate_split(selected=rows, experiments=experiments)
    assert len(experiments["A"]) == len(experiments["B"]) == 10
    assert Counter(row["category"] for row in experiments["A"]) == {
        f"category_{index}": 1 for index in range(10)
    }
    assert {
        row["prompt_id"] for row in experiments["A"]
    }.isdisjoint(row["prompt_id"] for row in experiments["B"])


def test_split_is_reproducible_and_balances_total_advantage():
    rows = candidates()
    first = split_category_pairs(rows)
    second = split_category_pairs(rows)
    assert [row["prompt_id"] for row in first["A"]] == [
        row["prompt_id"] for row in second["A"]
    ]
    total_a = sum(
        row["basis_advantage_over_best_control"] for row in first["A"]
    )
    total_b = sum(
        row["basis_advantage_over_best_control"] for row in first["B"]
    )
    assert total_a == pytest.approx(total_b)


def test_split_rejects_category_without_exactly_two_items():
    rows = candidates()[:-1]
    with pytest.raises(ValueError, match="各カテゴリ2件"):
        split_category_pairs(rows)
