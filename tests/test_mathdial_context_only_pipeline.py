"""MathDial context-only再学習・確認評価経路のテスト。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from core.dpo_prompting import (
    DPO_PROMPT_TEMPLATE_VERSION,
    NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION,
    NEUTRAL_CONVERSATION_INSTRUCTION,
    build_context_only_dpo_prompt,
    build_dpo_prompt,
    build_mathdial_dpo_prompt,
    build_neutral_conversation_dpo_prompt,
)
from tools.mathdial_evaluation import (
    blind_oracle_rows,
    build_mathdial_model_prompt,
    generate_three_model_responses,
    translate_prompts,
)
from tools.rewrite_mathdial_dpo_context_only import (
    PROMPT_REWRITE_VERSION,
    immutable_record_payload,
    rewrite_file,
    rewrite_record,
)
from tools.train_qwen35_dpo_lora import validate_tokenizer_prefix_alignment


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_GENERATION_TEXT = (
    "個別指導",
    "教師返答",
    "段階的ヒント",
    "理解確認",
    "equitable_tutoring",
    "BASiS",
)


def old_mathdial_prompt(index: int) -> str:
    return build_mathdial_dpo_prompt(
        history_turns=[
            {"speaker": "User", "text": f"{index} + 1を考えています。"},
            {"speaker": "AI", "text": "どのように考えましたか。"},
            {"speaker": "User", "text": f"{index + 1}だと思います。"},
        ]
    )


def dpo_record(index: int, *, gold: bool) -> dict[str, object]:
    source_dataset = "mathdial" if gold else "wildchat"
    return {
        "prompt": old_mathdial_prompt(index),
        "chosen": f"chosen-{index}",
        "rejected": f"rejected-{index}",
        "source_dataset": source_dataset,
        "source_dialogue_id": f"dialogue-{index}",
        "turn_index": index,
        "score_chosen": 0.9,
        "score_rejected": 0.2,
        "score_gap": 0.7,
        "acceptance_rule": "strict",
        "dpo_prompt_template_version": DPO_PROMPT_TEMPLATE_VERSION,
        "metadata": {
            "gold": gold,
            "source_dataset": source_dataset,
            "source_hash": f"source-{index}",
            "source_prompt_hash": f"prompt-{index}",
            "acceptance_rule": "strict",
            "dpo_prompt_template": DPO_PROMPT_TEMPLATE_VERSION,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def test_context_only_prompt_has_no_style_instruction_and_legacy_is_unchanged():
    history = [
        {"speaker": "User", "text": "2 + 3は何ですか。"},
        {"speaker": "AI", "text": "どう考えましたか。"},
        {"speaker": "User", "text": "6だと思います。"},
    ]
    context = build_context_only_dpo_prompt(history_turns=history)
    assert context == (
        "User: 2 + 3は何ですか。\n"
        "AI: どう考えましたか。\n"
        "User: 6だと思います。\n"
        "AI:"
    )
    assert not any(token in context for token in FORBIDDEN_GENERATION_TEXT)
    assert "以下の会話" not in context
    assert build_dpo_prompt(history_turns=history).startswith(
        "以下の会話の次のAI返答を生成してください。"
    )
    assert build_mathdial_dpo_prompt(history_turns=history).startswith(
        "以下の個別指導対話の次の教師返答を生成してください。"
    )
    with pytest.raises(ValueError, match="有効な会話履歴"):
        build_context_only_dpo_prompt()


def test_neutral_conversation_prompt_adds_only_minimum_task_instruction():
    history = [
        {"speaker": "User", "text": "2 + 3は何ですか。"},
        {"speaker": "AI", "text": "どう考えましたか。"},
        {"speaker": "User", "text": "6だと思います。"},
    ]
    prompt = build_neutral_conversation_dpo_prompt(history_turns=history)
    assert prompt == (
        f"{NEUTRAL_CONVERSATION_INSTRUCTION}\n\n"
        "User: 2 + 3は何ですか。\n"
        "AI: どう考えましたか。\n"
        "User: 6だと思います。\n"
        "AI:\n"
    )
    assert not any(token in prompt for token in FORBIDDEN_GENERATION_TEXT)
    assert "質問" not in prompt
    assert "ヒント" not in prompt
    assert "教師" not in prompt
    assert "数学" not in prompt
    assert "1〜2文" not in prompt


def test_rewrite_changes_only_prompt_fields_and_keeps_ordered_preferences(
    tmp_path: Path,
):
    source_rows = [dpo_record(index, gold=index >= 4) for index in range(6)]
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    write_jsonl(source, source_rows)

    summary = rewrite_file(
        source,
        output,
        expected_records=6,
        expected_gold=2,
        prompt_mode="neutral_conversation",
    )
    rewritten = read_jsonl(output)
    assert [row["chosen"] for row in rewritten] == [
        row["chosen"] for row in source_rows
    ]
    assert [row["rejected"] for row in rewritten] == [
        row["rejected"] for row in source_rows
    ]
    assert [
        immutable_record_payload(row) for row in rewritten
    ] == [immutable_record_payload(row) for row in source_rows]
    assert all(
        row["dpo_prompt_template_version"]
        == NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION
        for row in rewritten
    )
    assert all(
        str(row["prompt"]).startswith(
            NEUTRAL_CONVERSATION_INSTRUCTION + "\n\n"
        )
        for row in rewritten
    )
    for source_row, output_row in zip(source_rows, rewritten):
        assert output_row["metadata"]["frozen_chosen_sha256"] == hashlib.sha256(
            str(source_row["chosen"]).encode("utf-8")
        ).hexdigest()
        assert output_row["metadata"]["frozen_rejected_sha256"] == hashlib.sha256(
            str(source_row["rejected"]).encode("utf-8")
        ).hexdigest()
    assert all(str(row["prompt"]).endswith("AI:\n") for row in rewritten)
    assert all(
        not any(token in str(row["prompt"]) for token in FORBIDDEN_GENERATION_TEXT)
        for row in rewritten
    )
    assert summary["records"] == 6
    assert summary["gold_records"] == 2
    assert summary["local_prompt_mode"] == "neutral_conversation"
    assert summary["scores_acceptance_and_source_fields_unchanged"] is True


def test_rewrite_rejects_unknown_source_template():
    record = dpo_record(0, gold=False)
    record["dpo_prompt_template_version"] = "unknown"
    with pytest.raises(ValueError, match="旧prompt template"):
        rewrite_record(record, line_number=1)


def test_neutral_evaluation_prompt_keeps_problem_and_hides_references(
    tmp_path: Path,
):
    row = {
        "prompt_id": "p1",
        "sample_id": "s1",
        "problem_ja": "2 + 2を計算してください。",
        "ground_truth_ja": "4",
        "problem_en": "Compute 2 + 2.",
        "ground_truth_en": "4",
        "history_ja": [{"role": "user", "text": "5だと思います。"}],
        "history": [{"role": "user", "text": "5だと思います。"}],
    }
    prompt, version = build_mathdial_model_prompt(
        row,
        local_prompt_mode="neutral_conversation",
    )
    assert prompt.startswith(
        f"{NEUTRAL_CONVERSATION_INSTRUCTION}\n\n"
        "User: 2 + 2を計算してください。"
    )
    assert prompt.endswith("AI:\n")
    assert "5だと思います。" in prompt
    assert "ground_truth" not in prompt
    assert "Teacher" not in prompt
    assert "4" not in prompt
    assert not any(token in prompt for token in FORBIDDEN_GENERATION_TEXT)
    assert version == NEUTRAL_CONVERSATION_DPO_PROMPT_TEMPLATE_VERSION

    output = tmp_path / "responses.jsonl"
    generated = generate_three_model_responses(
        [row],
        base_model="base",
        basis_lora="basis",
        random_lora="random",
        output_path=output,
        mock=True,
        seed=42,
        local_prompt_mode="neutral_conversation",
    )
    assert generated[0]["model_prompt"] == prompt
    oracle = blind_oracle_rows(generated)
    assert len(oracle) == 3
    assert all(
        item["metadata"]["local_prompt_mode"] == "neutral_conversation"
        for item in oracle
    )
    assert all("basis" not in item["prompt"].lower() for item in oracle)


def test_tokenizer_prefix_gate_accepts_newline_and_rejects_merged_boundary():
    class BoundaryMergingTokenizer:
        def __call__(self, text: str) -> dict[str, list[int]]:
            ids = [ord(character) for character in text]
            marker = "AI:"
            marker_index = text.rfind(marker)
            if marker_index >= 0 and marker_index + len(marker) < len(text):
                following = text[marker_index + len(marker)]
                if following != "\n":
                    colon_index = marker_index + len(marker) - 1
                    ids[colon_index : colon_index + 2] = [1_000_000]
            return {"input_ids": ids}

    tokenizer = BoundaryMergingTokenizer()
    valid = {
        "prompt": "User: 質問です。\nAI:\n",
        "chosen": "応答です。",
        "rejected": "別の応答です。",
    }
    assert validate_tokenizer_prefix_alignment([valid], tokenizer) == {
        "records": 1,
        "chosen_mismatches": 0,
        "rejected_mismatches": 0,
    }

    invalid = dict(valid)
    invalid["prompt"] = "User: 質問です。\nAI:"
    with pytest.raises(ValueError, match="tokenizer境界"):
        validate_tokenizer_prefix_alignment([invalid], tokenizer)


def test_resume_rejects_mixed_prompt_modes(tmp_path: Path):
    existing = [
        {
            "prompt_id": "p1",
            "local_prompt_mode": "mathdial_instruction",
        }
    ]
    with pytest.raises(ValueError, match="local_prompt_mode"):
        translate_prompts(
            [],
            generator=None,
            model="mock",
            mock=True,
            existing=existing,
            local_prompt_mode="neutral_conversation",
        )

    output = tmp_path / "responses.jsonl"
    write_jsonl(output, existing)
    with pytest.raises(ValueError, match="local_prompt_mode"):
        generate_three_model_responses(
            [],
            base_model="base",
            basis_lora="basis",
            random_lora="random",
            output_path=output,
            mock=True,
            seed=42,
            local_prompt_mode="neutral_conversation",
        )


def create_pipeline_fixture(root: Path) -> None:
    basis = [dpo_record(index, gold=index >= 4) for index in range(6)]
    random_rows = [dpo_record(index + 10, gold=False) for index in range(6)]
    write_jsonl(root / "dpo/mathdial_basis_train.jsonl", basis)
    write_jsonl(root / "dpo/mathdial_random_train.jsonl", random_rows)

    conversations = []
    samples = []
    moves = ("probing", "focus", "telling", "generic")
    for index in range(9):
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
                "history": [
                    {
                        "role": "user",
                        "text": f"{index + 2}だと思います。",
                    }
                ],
                "metadata": {
                    "split": "test",
                    "history_ends_with_user": True,
                    "teacher_moves": [moves[index % len(moves)]],
                },
            }
        )
    write_jsonl(
        root / "mathdial/data/mathdial_conversations.jsonl",
        conversations,
    )
    write_jsonl(
        root / "mathdial/data/mathdial_assistant_samples.jsonl",
        samples,
    )
    write_jsonl(
        root / "evaluation/prompts_ja.jsonl",
        [{"sample_id": "sample-0", "qid": "qid-0"}],
    )


def test_neutral_prompt_pipeline_fixture_runs_without_api_or_gpu(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    create_pipeline_fixture(source)
    env = {
        **os.environ,
        "SOURCE_RUN": str(source),
        "RUN_TAG": "neutral_prompt_fixture",
        "OUTPUT_ROOT": str(output),
        "DRY_RUN": "1",
        "DRY_RUN_RECORDS_PER_ARM": "6",
        "DRY_RUN_BASIS_GOLD_RECORDS": "2",
        "DRY_RUN_EVAL_COUNT": "5",
        "START_STAGE": "rewrite_dpo",
        "END_STAGE": "report",
        "WORKERS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    completed = subprocess.run(
        ["bash", "scripts/run_mathdial_context_only_v2_pipeline.sh"],
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
    assert manifest["local_prompt_mode"] == "neutral_conversation"
    assert manifest["training"]["max_length"] == 4096
    assert manifest["training"]["device_map"] == "auto"
    assert manifest["training"]["max_memory"] == "0=38GiB,1=46GiB,cpu=0GiB"
    assert manifest["training"]["gpu0_minimum_activation_headroom_mib"] == 8192
    assert manifest["training"]["cuda_allocator_configuration"] == (
        "expandable_segments:True"
    )
    assert manifest["training"]["runtime_attempts"]
    assert manifest["training_data_policy"] == {
        "chosen_rejected": "unchanged_from_source_run",
        "basis_records": 6,
        "basis_selected_records": 4,
        "basis_gold_records": 2,
        "random_records": 6,
        "random_gold_records": 0,
    }
    prompts = read_jsonl(output / "evaluation/prompts_ja.jsonl")
    assert len(prompts) == 5
    assert not {"qid-0"} & {str(row["qid"]) for row in prompts}


def test_context_only_shell_scripts_have_valid_syntax():
    for relative in (
        "scripts/run_mathdial_context_only_v2_pipeline.sh",
        "scripts/run_mathdial_context_only_v2_watchdog.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)


def test_context_only_training_reserves_gpu0_headroom_and_bounds_oom_retry():
    pipeline = (
        ROOT / "scripts/run_mathdial_context_only_v2_pipeline.sh"
    ).read_text(encoding="utf-8")
    watchdog = (
        ROOT / "scripts/run_mathdial_context_only_v2_watchdog.sh"
    ).read_text(encoding="utf-8")

    assert (
        'TRAIN_MAX_MEMORY="${TRAIN_MAX_MEMORY:-0=38GiB,1=46GiB,cpu=0GiB}"'
        in pipeline
    )
    assert 'TRAIN_GPU0_MIN_HEADROOM_MIB="${TRAIN_GPU0_MIN_HEADROOM_MIB:-8192}"' in pipeline
    assert "train_placement_preflight" in pipeline
    assert "GPU 0のactivation用余白が不足しています" in pipeline
    assert "OOM_DETECTED.json" in pipeline
    assert (
        'WATCHDOG_OOM_TRAIN_MAX_MEMORY="${WATCHDOG_OOM_TRAIN_MAX_MEMORY:-0=36GiB,1=46GiB,cpu=0GiB}"'
        in watchdog
    )
    assert "oom_fallback_used=0" in watchdog
    assert "headroom拡大後もCUDA OOMが再発したため、安全に停止します" in watchdog
    assert "ALLOW_TRAIN_PLACEMENT_CONTINUATION=1" in watchdog


def test_placement_only_continuation_preserves_research_conditions(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    create_pipeline_fixture(source)
    base_env = {
        **os.environ,
        "SOURCE_RUN": str(source),
        "RUN_TAG": "placement_continuation_fixture",
        "OUTPUT_ROOT": str(output),
        "DRY_RUN": "1",
        "DRY_RUN_RECORDS_PER_ARM": "6",
        "DRY_RUN_BASIS_GOLD_RECORDS": "2",
        "DRY_RUN_EVAL_COUNT": "5",
        "START_STAGE": "rewrite_dpo",
        "END_STAGE": "rewrite_dpo",
        "PYTHONUNBUFFERED": "1",
    }
    initial = subprocess.run(
        ["bash", "scripts/run_mathdial_context_only_v2_pipeline.sh"],
        cwd=ROOT,
        env=base_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert initial.returncode == 0, initial.stdout + initial.stderr

    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["experiment_fingerprint"] = "legacy-fixture-fingerprint"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    continued = subprocess.run(
        ["bash", "scripts/run_mathdial_context_only_v2_pipeline.sh"],
        cwd=ROOT,
        env={
            **base_env,
            "ALLOW_TRAIN_PLACEMENT_CONTINUATION": "1",
            "TRAIN_MAX_MEMORY": "0=38GiB,1=46GiB,cpu=0GiB",
            "FORCE_STAGE": "rewrite_dpo",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert continued.returncode == 0, continued.stdout + continued.stderr
    attempts = read_jsonl(output / "training_runtime_attempts.jsonl")
    assert attempts[-1]["placement_only_continuation"] is True
    assert attempts[-1]["experiment_fingerprint"] == "legacy-fixture-fingerprint"

    changed_research_condition = subprocess.run(
        ["bash", "scripts/run_mathdial_context_only_v2_pipeline.sh"],
        cwd=ROOT,
        env={
            **base_env,
            "ALLOW_TRAIN_PLACEMENT_CONTINUATION": "1",
            "TRAIN_MAX_LENGTH": "2048",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert changed_research_condition.returncode != 0
    assert "max_lengthが変わっているため継続できません" in (
        changed_research_condition.stdout + changed_research_condition.stderr
    )
