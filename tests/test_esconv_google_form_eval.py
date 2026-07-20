from __future__ import annotations

from collections import Counter

from scripts.prepare_esconv_google_form_eval import (
    MODEL_KEYS,
    QUESTIONS,
    public_record,
    select_model_blind,
    select_oracle_enriched,
    version_orders,
)


def candidate(index: int, category: str, advantage: float) -> dict:
    return {
        "prompt_id": f"p{index:03d}",
        "category": category,
        "axis_focus": [],
        "history": [{"speaker": "User", "text": f"履歴{index}"}],
        "prompt": f"相談{index}",
        "responses": {
            "base": f"base {index}",
            "basis": f"basis {index}",
            "random": f"random {index}",
        },
        "oracle_axis_scores": {},
        "representative_means": {
            "base": 7.0,
            "basis": 8.0 + advantage,
            "random": 7.5,
        },
        "basis_advantage_over_best_control": advantage,
    }


def make_candidates() -> list[dict]:
    rows = []
    for category_index in range(10):
        category = f"category_{category_index}"
        for item_index in range(10):
            index = category_index * 10 + item_index
            rows.append(candidate(index, category, (item_index + 1) / 10))
    return rows


def test_model_blind_selection_is_reproducible_and_balanced():
    rows = make_candidates()
    first = select_model_blind(rows, total=20, seed=42)
    second = select_model_blind(rows, total=20, seed=42)
    assert [row["prompt_id"] for row in first] == [
        row["prompt_id"] for row in second
    ]
    assert Counter(row["category"] for row in first) == {
        f"category_{index}": 2 for index in range(10)
    }


def test_oracle_enriched_selection_uses_positive_top_items():
    selected = select_oracle_enriched(make_candidates(), total=20)
    assert len(selected) == 20
    assert all(row["basis_advantage_over_best_control"] > 0 for row in selected)
    by_category = {}
    for row in selected:
        by_category.setdefault(row["category"], []).append(
            row["basis_advantage_over_best_control"]
        )
    assert all(sorted(values) == [0.9, 1.0] for values in by_category.values())


def test_three_form_versions_rotate_every_model_through_every_position():
    orders = version_orders(20, seed=42)
    for item_index in range(20):
        position_models = {
            position: {
                orders[version][item_index][position_index]
                for version in ("A", "B", "C")
            }
            for position_index, position in enumerate(("A", "B", "C"))
        }
        assert all(models == set(MODEL_KEYS) for models in position_models.values())


def test_public_record_is_blind_and_has_seven_plain_questions():
    row = candidate(1, "category", 0.5)
    public = public_record(
        row,
        item_number=1,
        order=("basis", "base", "random"),
    )
    serialized = str(public)
    assert "bayes_dpo" not in serialized
    assert "Random-DPO" not in serialized
    assert "oracle" not in serialized.lower()
    assert len(public["questions"]) == len(QUESTIONS) == 7
    assert public["response_a"] == "basis 1"
    assert public["conversation"].endswith("相談者: 相談1")
