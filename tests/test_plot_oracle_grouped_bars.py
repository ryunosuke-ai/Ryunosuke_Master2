from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.plot_oracle_grouped_bars import (
    AxisSpec,
    FigureSpec,
    analyze_figure,
    render_figure,
)


def write_raw(path: Path, *, basis_offset: float = 2.0) -> None:
    rows = []
    for index in range(12):
        sample_id = f"sample_{index:02d}"
        base = 4.0 + (index % 2)
        random = 4.0 + ((index + 1) % 2)
        for model, score in (
            ("base", base),
            ("basis", base + basis_offset),
            ("random_dpo", random),
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "model_name": model,
                    "scores": {"axis": score},
                }
            )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def fixture_spec(raw_path: Path, basis_mean: float = 6.5) -> FigureSpec:
    return FigureSpec(
        slug="fixture",
        title="Fixture",
        axes=(
            AxisSpec(
                key="axis",
                label="Test Axis",
                means={
                    "base": 4.5,
                    "basis": basis_mean,
                    "random_dpo": 4.5,
                },
                raw_path=raw_path,
            ),
        ),
    )


def test_analysis_validates_means_and_computes_significance(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    write_raw(raw)
    summary, significance = analyze_figure(
        fixture_spec(raw),
        bootstrap=200,
        permutations=500,
        seed=42,
    )
    assert len(summary) == 3
    assert all(row["n"] == 12 for row in summary)
    comparisons = {
        row["comparison"]: row for row in significance
    }
    assert comparisons["BASiS vs Base"]["significant"]
    assert comparisons["BASiS vs Random-DPO"]["significant"]
    assert not comparisons["Base vs Random-DPO"]["significant"]


def test_analysis_rejects_supplied_mean_mismatch(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    write_raw(raw)
    with pytest.raises(ValueError, match="指定平均"):
        analyze_figure(
            fixture_spec(raw, basis_mean=8.0),
            bootstrap=100,
            permutations=100,
            seed=42,
        )


def test_render_writes_vector_and_raster_outputs(tmp_path: Path):
    pytest.importorskip("matplotlib")
    raw = tmp_path / "raw.jsonl"
    write_raw(raw)
    spec = fixture_spec(raw)
    summary, significance = analyze_figure(
        spec,
        bootstrap=100,
        permutations=200,
        seed=42,
    )
    outputs = render_figure(
        spec,
        summary,
        significance,
        output_dir=tmp_path / "figures",
        dpi=80,
        y_min=0.0,
    )
    assert all(path.stat().st_size > 100 for path in outputs)
