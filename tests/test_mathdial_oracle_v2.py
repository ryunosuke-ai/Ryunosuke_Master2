"""MathDial v2確認評価の軽量テスト。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.eval_oracle_mathdial_v2 import CORE_V2
from tools.mathdial_evaluation import (
    exclusion_ids_from_prompts,
    select_test_prompts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_v2_uses_paper_grounded_conditional_axes():
    assert [axis.key for axis in CORE_V2.axes] == [
        "equitable_tutoring",
        "learner_reasoning_diagnosis",
        "mistake_location_and_targeting",
        "guidance_quality",
        "feedback_actionability",
        "answer_revealing_calibration",
        "teacher_move_stage_alignment",
    ]
    text = " ".join(
        axis.description + axis.high + axis.low + axis.ten_point_guidance
        for axis in CORE_V2.axes
    ).lower()
    assert "常に答えを隠すことを高評価にはしない" in text
    assert "正答に存在しない誤り" in text
    assert "probing" in text
    assert "focus" in text
    assert "telling" in text
    assert "generic" in text
    assert "basis-dpo" not in text
    assert "random-dpo" not in text


def test_confirmation_prompt_selection_excludes_v1_and_stratifies_moves(
    tmp_path: Path,
):
    conversations = []
    samples = []
    moves = ("probing", "focus", "telling", "generic")
    for index in range(12):
        conversation_id = f"c{index}"
        conversations.append(
            {
                "conversation_id": conversation_id,
                "split": "test",
                "metadata": {
                    "qid": f"q{index}",
                    "question": f"question {index}",
                    "ground_truth": str(index),
                },
            }
        )
        samples.append(
            {
                "sample_id": f"s{index}",
                "conversation_id": conversation_id,
                "history": [{"role": "user", "text": str(index)}],
                "metadata": {
                    "split": "test",
                    "history_ends_with_user": True,
                    "teacher_moves": [moves[index % len(moves)]],
                },
            }
        )
    previous = tmp_path / "previous.jsonl"
    previous.write_text(
        json.dumps({"sample_id": "s0", "qid": "q0"}) + "\n",
        encoding="utf-8",
    )
    excluded_samples, excluded_qids = exclusion_ids_from_prompts([previous])
    selected = select_test_prompts(
        samples,
        conversations,
        count=8,
        seed=42,
        excluded_sample_ids=excluded_samples,
        excluded_qids=excluded_qids,
        stratify_teacher_moves=True,
        prompt_id_prefix="confirm",
    )
    assert len(selected) == 8
    assert all(row["sample_id"] != "s0" and row["qid"] != "q0" for row in selected)
    assert len({row["qid"] for row in selected}) == 8
    assert all(row["prompt_id"].startswith("confirm_") for row in selected)


def test_mathdial_oracle_v2_shell_syntax():
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/run_mathdial_oracle_v2_confirmation.sh")],
        check=True,
    )
