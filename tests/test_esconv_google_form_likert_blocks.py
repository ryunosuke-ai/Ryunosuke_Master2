from __future__ import annotations

from collections import Counter

import pytest

from scripts.prepare_esconv_google_form_likert_blocks import (
    balanced_single_form_orders,
    select_discriminative_items,
    split_discriminative_items,
    split_category_pairs,
    validate_split,
)


def candidate(index: int, category: str, advantage: float) -> dict:
    return {
        "prompt_id": f"p{index:03d}",
        "category": category,
        "basis_advantage_over_best_control": advantage,
        "representative_means": {
            "base": 6.0,
            "basis": 9.0,
            "random": 6.5,
        },
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


def test_single_form_orders_balance_every_model_across_positions():
    orders = balanced_single_form_orders(10, seed=42)
    counts = Counter(
        (position, model)
        for order in orders
        for position, model in zip(("A", "B", "C"), order)
    )
    assert len(orders) == 10
    for model in ("base", "basis", "random"):
        assert sorted(counts[(position, model)] for position in ("A", "B", "C")) == [
            3,
            3,
            4,
        ]


def test_discriminative_selection_uses_top_advantages():
    rows = [
        candidate(index, f"category_{index % 5}", index / 10)
        for index in range(30)
    ]
    selected = select_discriminative_items(rows, total=20)
    assert len(selected) == 20
    assert min(row["basis_advantage_over_best_control"] for row in selected) == 1.0


def test_discriminative_split_is_disjoint_complete_and_balanced():
    rows = [
        candidate(index, f"category_{index % 5}", 0.6 + index / 10)
        for index in range(20)
    ]
    experiments = split_discriminative_items(rows)
    ids_a = {row["prompt_id"] for row in experiments["A"]}
    ids_b = {row["prompt_id"] for row in experiments["B"]}
    assert len(ids_a) == len(ids_b) == 10
    assert not ids_a & ids_b
    assert ids_a | ids_b == {row["prompt_id"] for row in rows}
