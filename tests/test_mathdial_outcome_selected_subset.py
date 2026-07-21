from tools.build_mathdial_outcome_selected_subset import select_sample_ids


def test_select_sample_ids_orders_by_basis_margin_deterministically() -> None:
    rows = []
    values = {
        "s1": {"base": 5, "basis": 8, "random_dpo": 6},
        "s2": {"base": 7, "basis": 8, "random_dpo": 6},
        "s3": {"base": 4, "basis": 7, "random_dpo": 5},
    }
    for sample_id, scores in values.items():
        for model, score in scores.items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "model_name": model,
                    "overall_score": score,
                }
            )

    selected = select_sample_ids(rows, 2)

    # s1とs3は同marginなので、sample_id降順で固定される。
    assert [row["sample_id"] for row in selected] == ["s3", "s1"]
    assert [row["selection_rank"] for row in selected] == [1, 2]
    assert all(row["selection_margin"] == 2 for row in selected)


def test_select_sample_ids_requires_complete_three_model_scores() -> None:
    rows = [
        {"sample_id": "s1", "model_name": "base", "overall_score": 5},
        {"sample_id": "s1", "model_name": "basis", "overall_score": 8},
    ]

    try:
        select_sample_ids(rows, 1)
    except ValueError as exc:
        assert "完全な3モデル評価が不足" in str(exc)
    else:
        raise AssertionError("不完全な3モデル評価を受理してはいけない")
