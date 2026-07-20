"""MathDial旧instruction・識別力重視追試のテスト。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts.run_mathdial_statistics import analyze_strata
from tools.mathdial_evaluation import (
    DISCRIMINATIVE_SAMPLING_VERSION,
    build_mathdial_model_prompt,
    finalize_discriminative_translations,
    load_discriminative_quota_config,
    select_discriminative_followup_prompts,
)


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def history_for_stage(stage: str, marker: int) -> list[dict[str, str]]:
    lengths = {"initial": 2, "guided": 4, "advanced": 10}
    rows = []
    for index in range(lengths[stage]):
        role = "assistant" if index % 2 == 0 else "user"
        rows.append(
            {
                "role": role,
                "text": (
                    f"途中の説明 {marker}-{index}"
                    if role == "assistant"
                    else f"{marker} + {index} = {marker + index} と考えました。"
                ),
            }
        )
    return rows


def discriminative_fixture(
    *,
    per_stratum: int,
) -> tuple[list[dict], list[dict]]:
    samples = []
    conversations = []
    index = 0
    for move in ("probing", "telling", "focus"):
        for stage in ("initial", "guided", "advanced"):
            for _ in range(per_stratum):
                conversation_id = f"conversation-{index}"
                conversations.append(
                    {
                        "conversation_id": conversation_id,
                        "split": "test",
                        "metadata": {
                            "qid": f"qid-{index}",
                            "question": f"{index} + 1はいくつですか。",
                            "ground_truth": str(index + 1),
                        },
                    }
                )
                samples.append(
                    {
                        "sample_id": f"sample-{index}",
                        "conversation_id": conversation_id,
                        "history": history_for_stage(stage, index),
                        "response": "source teacher response",
                        "next_user_turn": "もう一度考えます。",
                        "metadata": {
                            "split": "test",
                            "history_ends_with_user": True,
                            "teacher_moves": [move],
                        },
                    }
                )
                index += 1
    return samples, conversations


def test_default_quota_config_is_fixed_to_150_plus_18_reserves():
    quotas, reserve, config = load_discriminative_quota_config(
        ROOT
        / "configs/evaluations/mathdial_discriminative_followup_v1.yaml"
    )
    assert config["version"] == DISCRIMINATIVE_SAMPLING_VERSION
    assert sum(quotas.values()) == 150
    assert reserve == 2
    assert quotas == {
        ("probing", "initial"): 15,
        ("probing", "guided"): 25,
        ("probing", "advanced"): 20,
        ("telling", "initial"): 15,
        ("telling", "guided"): 20,
        ("telling", "advanced"): 15,
        ("focus", "initial"): 10,
        ("focus", "guided"): 15,
        ("focus", "advanced"): 15,
    }


def test_discriminative_selection_is_deterministic_and_uses_reserves():
    samples, conversations = discriminative_fixture(per_stratum=3)
    quotas = {
        (move, stage): 2
        for move in ("probing", "telling", "focus")
        for stage in ("initial", "guided", "advanced")
    }
    first, manifest = select_discriminative_followup_prompts(
        samples,
        conversations,
        quotas=quotas,
        reserve_per_stratum=1,
        seed=42,
        excluded_qids=set(),
    )
    second, _ = select_discriminative_followup_prompts(
        samples,
        conversations,
        quotas=quotas,
        reserve_per_stratum=1,
        seed=42,
        excluded_qids=set(),
    )
    assert [row["sample_id"] for row in first] == [
        row["sample_id"] for row in second
    ]
    assert len(first) == len({row["qid"] for row in first}) == 27
    assert manifest["candidate_count"] == 27
    assert all(row["next_user_turn"] for row in samples)
    assert all(
        row["selection_teacher_move"] != "generic" for row in first
    )

    failed_primary = {
        next(
            row["sample_id"]
            for row in first
            if row["selection_teacher_move"] == move
            and row["selection_stage"] == stage
            and row["selection_role"] == "primary"
        )
        for move, stage in quotas
    }
    translated = [
        {
            **row,
            "problem_ja": row["problem_en"],
            "ground_truth_ja": row["ground_truth_en"],
            "history_ja": row["history_en"],
        }
        for row in first
        if row["sample_id"] not in failed_primary
    ]
    final = finalize_discriminative_translations(
        translated,
        quotas=quotas,
    )
    assert len(final) == 18
    assert sum(row["selection_role"] == "reserve" for row in final) == 9


def test_old_instruction_prompt_is_exact_and_does_not_expose_audit_labels():
    row = {
        "problem_ja": "2 + 2はいくつですか。",
        "history_ja": [
            {"role": "user", "text": "5だと思います。"},
        ],
        "ground_truth_ja": "4",
        "selection_teacher_move": "probing",
        "selection_stage": "initial",
    }
    prompt, version = build_mathdial_model_prompt(
        row,
        local_prompt_mode="mathdial_instruction",
    )
    assert version == "dpo_user_ai_instruction.v1"
    assert prompt.startswith(
        "以下の個別指導対話の次の教師返答を生成してください。\n"
        "返答は自然な日本語で、問題とこれまでの学習者の考えに即して"
        "簡潔に書いてください。\n"
        "必要に応じて質問、焦点化、段階的ヒント、説明、理解確認の"
        "いずれかを選んでください。"
    )
    assert prompt.endswith("AI:")
    assert not prompt.endswith("AI:\n")
    assert "probing" not in prompt
    assert "initial" not in prompt
    assert "\n4\n" not in prompt

    long_row = {
        **row,
        "history_ja": [
            {
                "role": "assistant" if index % 2 == 0 else "user",
                "text": f"履歴{index}",
            }
            for index in range(14)
        ],
    }
    long_prompt, _ = build_mathdial_model_prompt(
        long_row,
        local_prompt_mode="mathdial_instruction",
    )
    assert "数学問題: 2 + 2はいくつですか。" in long_prompt
    assert "履歴0" not in long_prompt
    assert "履歴13" in long_prompt


def test_stratum_statistics_are_descriptive_only():
    data = {
        "pedagogical_v2.category_overall": {
            "s1": {"base": 5.0, "basis": 7.0, "random_dpo": 6.0},
            "s2": {"base": 6.0, "basis": 8.0, "random_dpo": 5.0},
        }
    }
    strata = {
        "s1": {"teacher_move": "probing", "conversation_stage": "initial"},
        "s2": {"teacher_move": "telling", "conversation_stage": "advanced"},
    }
    summaries, pairwise = analyze_strata(
        data,
        strata,
        bootstrap=20,
        seed=42,
    )
    assert summaries
    assert pairwise
    assert {
        row["inference_status"] for row in pairwise
    } == {"exploratory_descriptive_only"}


def test_instruction_discriminative_pipeline_fixture_runs(tmp_path: Path):
    source = tmp_path / "source"
    neutral = tmp_path / "neutral"
    output = tmp_path / "output"
    samples, conversations = discriminative_fixture(per_stratum=1)
    write_jsonl(
        source / "mathdial/data/mathdial_assistant_samples.jsonl",
        samples,
    )
    write_jsonl(
        source / "mathdial/data/mathdial_conversations.jsonl",
        conversations,
    )
    write_jsonl(
        source / "evaluation/prompts_ja.jsonl",
        [{"sample_id": "old-v6", "qid": "old-v6"}],
    )
    write_jsonl(
        neutral / "evaluation/prompts_ja.jsonl",
        [{"sample_id": "old-v11", "qid": "old-v11"}],
    )
    for arm in ("basis_lora", "random_lora"):
        directory = source / "training" / arm
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (directory / "adapter_model.safetensors").write_bytes(b"dry-run")
    quota = tmp_path / "quota.yaml"
    quota.write_text(
        "\n".join(
            [
                "name: fixture",
                f"version: {DISCRIMINATIVE_SAMPLING_VERSION}",
                "target_count: 9",
                "reserve_per_stratum: 0",
                "quotas:",
                "  probing: {initial: 1, guided: 1, advanced: 1}",
                "  telling: {initial: 1, guided: 1, advanced: 1}",
                "  focus: {initial: 1, guided: 1, advanced: 1}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "SOURCE_RUN": str(source),
        "EXCLUDE_NEUTRAL_RUN": str(neutral),
        "OUTPUT_ROOT": str(output),
        "RUN_TAG": "fixture",
        "EVAL_COUNT": "9",
        "MATHDIAL_DISCRIMINATIVE_QUOTA_CONFIG": str(quota),
        "DRY_RUN": "1",
        "WORKERS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_mathdial_instruction_discriminative_v2_pipeline.sh",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output / "report.md").is_file()
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evaluation_count"] == 9
    assert manifest["adapter_policy"] == "reuse_v6_without_retraining"


def test_instruction_discriminative_shell_syntax():
    for relative in (
        "scripts/run_mathdial_instruction_discriminative_v2_pipeline.sh",
        "scripts/run_mathdial_instruction_discriminative_v2_watchdog.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)
