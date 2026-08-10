from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.export_gold_only_axis_results import (
    REPRESENTATIVE_AXES,
    load_scores,
    select_representative_scores,
    write_scores,
)
from scripts.run_gold_only_four_model_statistics import analyze, load_axis_scores
from tools.gold_only_dpo import (
    FOUR_MODELS,
    audit_evaluation_leakage,
    build_oracle_input,
    generate_gold_responses,
    merge_oracle_raw,
    prepare_data,
)
from tools import run_oracle_evaluation_lora_pair as lora_pair


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def preference(index: int) -> dict:
    return {
        "prompt": f"User: p{index}\nAI:",
        "chosen": f"chosen {index}",
        "rejected": f"rejected {index}",
        "source_dataset": "Fixture",
        "source_dialogue_id": f"train-{index}",
        "turn_index": index,
        "metadata": {"source_split": "train"},
    }


def fixture_config(tmp_path: Path) -> Path:
    gold = [preference(index) for index in range(2)]
    gold_path = tmp_path / "gold.jsonl"
    basis_path = tmp_path / "basis.jsonl"
    eval_path = tmp_path / "evaluation.jsonl"
    template_path = tmp_path / "oracle_template.jsonl"
    write_jsonl(gold_path, gold)
    write_jsonl(basis_path, [*gold, preference(3)])
    evaluation = [
        {
            "sample_id": f"test-{index}",
            "conversation_id": f"test-conversation-{index}",
            "split": "test",
            "model_prompt": f"User: evaluation {index}\nAI:",
            "prompt": f"evaluation {index}",
            "history": [],
        }
        for index in range(2)
    ]
    write_jsonl(eval_path, evaluation)
    write_jsonl(
        template_path,
        [
            {
                "sample_id": row["sample_id"],
                "model_name": model,
                "prompt": row["prompt"],
                "history": [],
                "response": model,
                "metadata": {"conversation_id": row["conversation_id"]},
            }
            for row in evaluation
            for model in ("base", "basis", "random_dpo")
        ],
    )
    config = {
        "version": "gold_only_dpo500.v1",
        "expected_gold_records": 2,
        "training": {"gradient_accumulation_steps": 1},
        "datasets": {
            "mathdial": {
                "gold_source": str(gold_path),
                "basis_train_source": str(basis_path),
                "evaluation_source": str(eval_path),
                "oracle_template_source": str(template_path),
                "expected_eval_records": 2,
                "generation": {
                    "max_new_tokens": 32,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.05,
                },
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_prepare_data_copies_gold_byte_exact_and_audits_membership(tmp_path: Path):
    config = fixture_config(tmp_path)
    output = tmp_path / "run/train.jsonl"
    manifest_path = tmp_path / "run/manifest.json"
    manifest = prepare_data(
        config_path=config,
        dataset="mathdial",
        output=output,
        manifest_path=manifest_path,
    )
    source = tmp_path / "gold.jsonl"
    assert output.read_bytes() == source.read_bytes()
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["audit"]["basis_membership_verified"] is True
    assert manifest["audit"]["evaluation_leakage"]["conversation_id_overlap"] == 0


def test_evaluation_conversation_leakage_is_fatal():
    with pytest.raises(ValueError, match="会話ID重複"):
        audit_evaluation_leakage(
            [{"source_dialogue_id": "same"}],
            [{"conversation_id": "same", "split": "test"}],
            dataset="mathdial",
        )


def test_mock_generation_resume_and_oracle_prompt_hash(tmp_path: Path):
    config = fixture_config(tmp_path)
    responses = tmp_path / "responses.jsonl"
    first = generate_gold_responses(
        config_path=config,
        dataset="mathdial",
        lora_path=tmp_path / "unused",
        output=responses,
        base_model="fixture",
        seed=42,
        mock=True,
    )
    second = generate_gold_responses(
        config_path=config,
        dataset="mathdial",
        lora_path=tmp_path / "unused",
        output=responses,
        base_model="fixture",
        seed=42,
        mock=True,
    )
    assert first == second
    assert len(first) == 2
    assert all(row["model_prompt_sha256"] for row in first)
    oracle = tmp_path / "oracle.jsonl"
    main, ood = build_oracle_input(
        config_path=config,
        dataset="mathdial",
        responses_path=responses,
        output=oracle,
        ood_output=None,
    )
    assert len(main) == 2 and ood == []
    assert {row["model_name"] for row in main} == {"gold_only"}
    assert all(row["metadata"]["model_prompt_sha256"] for row in main)


def oracle_row(sample: str, model: str, value: int = 8) -> dict:
    return {
        "sample_id": sample,
        "model_name": model,
        "scores": {"axis": value},
        "overall_score": value,
        "judge_model": "judge",
        "oracle_prompt_version": "rubric.v1",
        "oracle_eval_category": "category",
        "score_scale": 10,
        "score_min": 1,
        "score_max": 10,
    }


def test_merge_raw_requires_exact_four_model_coverage_and_preserves_existing(tmp_path: Path):
    existing_path = tmp_path / "existing.jsonl"
    gold_path = tmp_path / "gold_raw.jsonl"
    existing = [
        oracle_row(sample, model)
        for sample in ("s1", "s2")
        for model in ("base", "bayes_dpo", "random_dpo")
    ]
    write_jsonl(existing_path, existing)
    write_jsonl(gold_path, [oracle_row(sample, "gold_only") for sample in ("s1", "s2")])
    original_hash = hashlib.sha256(existing_path.read_bytes()).hexdigest()
    payload = merge_oracle_raw(
        existing_path=existing_path,
        gold_path=gold_path,
        output=tmp_path / "combined.jsonl",
        manifest_path=tmp_path / "combined.manifest.json",
        expected_samples=2,
    )
    assert payload["records"] == 8
    assert hashlib.sha256(existing_path.read_bytes()).hexdigest() == original_hash
    assert set(FOUR_MODELS) == {"base", "basis", "random_dpo", "gold_only"}


def test_merge_raw_rejects_oracle_signature_mismatch(tmp_path: Path):
    existing_path = tmp_path / "existing.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    write_jsonl(
        existing_path,
        [oracle_row("s1", model) for model in ("base", "bayes_dpo", "random_dpo")],
    )
    mismatched = oracle_row("s1", "gold_only")
    mismatched["judge_model"] = "different"
    write_jsonl(gold_path, [mismatched])
    with pytest.raises(ValueError, match="Oracle条件"):
        merge_oracle_raw(
            existing_path=existing_path,
            gold_path=gold_path,
            output=tmp_path / "combined.jsonl",
            manifest_path=tmp_path / "manifest.json",
            expected_samples=1,
        )


def test_four_model_friedman_and_six_holm_pairs():
    data = {
        "category.axis": {
            f"sample-{index}": {
                "base": 4.0,
                "basis": 9.0,
                "random_dpo": 5.0,
                "gold_only": 7.0,
            }
            for index in range(12)
        }
    }
    summary, omnibus, posthoc = analyze(
        data, permutations=200, bootstrap=100, seed=42
    )
    assert len(summary) == 4
    assert omnibus[0]["models"] == 4
    assert omnibus[0]["significant"] is True
    assert len(posthoc) == 6
    assert {row["comparison"] for row in posthoc} == {
        "BASiS_vs_Base",
        "BASiS_vs_Random-DPO",
        "BASiS_vs_Gold-only",
        "Gold-only_vs_Base",
        "Gold-only_vs_Random-DPO",
        "Base_vs_Random-DPO",
    }


def test_four_model_statistics_rejects_incomplete_sample(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [oracle_row("s1", model) for model in ("base", "basis", "random_dpo")])
    data = load_axis_scores([("category", raw)])
    with pytest.raises(ValueError, match="4モデルが揃わない"):
        analyze(data, permutations=10, bootstrap=10, seed=42)


def test_lora_pair_loader_keeps_existing_two_adapter_contract(monkeypatch):
    captured = {}

    def fake_loader(base_model_id, *, adapters, use_4bit):
        captured.update(
            base_model_id=base_model_id, adapters=adapters, use_4bit=use_4bit
        )
        return "bundle"

    monkeypatch.setattr(lora_pair, "load_lora_bundle", fake_loader)
    result = lora_pair.load_lora_pair_bundle(
        "base", base_lora_path="basis", dpo_lora_path="random", use_4bit=False
    )
    assert result == "bundle"
    assert captured["adapters"] == {"bayes_dpo": "basis", "random_dpo": "random"}


def test_gold_only_shell_scripts_have_valid_syntax_and_fixed_conditions():
    for relative in (
        "scripts/run_gold_only_dpo_dataset_pipeline.sh",
        "scripts/run_gold_only_dpo_dataset_watchdog.sh",
        "scripts/run_gold_only_dpo_all_watchdog.sh",
        "scripts/complete_gold_only_representative_axes.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / relative)], check=True)
    pipeline = (ROOT / "scripts/run_gold_only_dpo_dataset_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert "--num-train-epochs 1" in pipeline
    assert "--max-length 1024" in pipeline
    assert "--no-4bit" in pipeline
    assert "--resume-from-checkpoint auto" in pipeline
    assert "START_STAGE" in pipeline and "END_STAGE" in pipeline


def test_representative_axis_selection_is_exact_and_ordered():
    for dataset, axes in REPRESENTATIVE_AXES.items():
        rows = [
            {"axis_key": axis, "value": index}
            for index, axis in enumerate(reversed(axes))
        ]
        selected = select_representative_scores(dataset, rows)
        assert [row["axis_key"] for row in selected] == list(axes)


def test_representative_axis_selection_rejects_missing_axis():
    axes = REPRESENTATIVE_AXES["esconv"]
    rows = [{"axis_key": axis} for axis in axes[:-1]]
    with pytest.raises(ValueError, match="代表軸の評価が不足"):
        select_representative_scores("esconv", rows)


def test_gold_only_watchdog_restarts_stalled_resumable_stage(tmp_path: Path):
    fake = tmp_path / "fake_pipeline.sh"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$OUTPUT_ROOT\"\n"
        "printf '{\"stage\":\"generate_responses\"}\\n' > \"$OUTPUT_ROOT/pipeline_status.json\"\n"
        "if [[ \"${WATCHDOG_ATTEMPT:-1}\" == \"1\" ]]; then sleep 30; fi\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = {
        "PATH": "/usr/bin:/bin",
        "DATASET": "esconv",
        "RUN_TAG": "watchdog-fixture",
        "OUTPUT_ROOT": str(tmp_path / "run"),
        "GOLD_ONLY_PIPELINE_SCRIPT": str(fake),
        "WATCHDOG_INTERVAL_SECONDS": "1",
        "WATCHDOG_STALL_SECONDS": "1",
        "WATCHDOG_MAX_RESTARTS": "2",
        "WATCHDOG_RESTART_DELAY_SECONDS": "0",
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/run_gold_only_dpo_dataset_watchdog.sh")],
        cwd=ROOT,
        env=env,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0
    log = (tmp_path / "run/watchdog/watchdog.log").read_text(encoding="utf-8")
    assert "stall dataset=esconv stage=generate_responses" in log
    assert "attempt=2" in log


def test_axis_score_export_writes_readable_text_and_json(tmp_path: Path):
    statistics = tmp_path / "statistics"
    statistics.mkdir()
    rows = [
        {
            "axis": "style.axis_one",
            "model_name": model,
            "n": "10",
            "mean": str(value),
            "std": "0.5",
            "ci95_low": str(value - 0.2),
            "ci95_high": str(value + 0.2),
            "is_highest": str(model == "basis"),
        }
        for model, value in zip(
            ("base", "gold_only", "basis", "random_dpo"),
            (6.0, 7.0, 8.0, 5.0),
        )
    ]
    summary_path = statistics / "model_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    omnibus_rows = [
        {
            "axis": "style.axis_one",
            "n": "10",
            "models": "4",
            "friedman_chi2": "9.0",
            "degrees_of_freedom": "3",
            "p_value": "0.01",
            "kendalls_w": "0.3",
            "significant": "True",
            "highest_model": "basis",
        }
    ]
    with (statistics / "omnibus_friedman.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(omnibus_rows[0]))
        writer.writeheader()
        writer.writerows(omnibus_rows)
    posthoc_rows = [
        {
            "axis": "style.axis_one",
            "comparison": "BASiS_vs_Base",
            "n": "10",
            "mean_diff": "2.0",
            "median_diff": "2.0",
            "ci95_low": "1.0",
            "ci95_high": "3.0",
            "p_raw": "0.0005",
            "p_holm": "0.0008",
            "cohens_dz": "1.0",
            "rank_biserial": "1.0",
            "wins": "10",
            "ties": "0",
            "losses": "0",
            "left_win_rate": "1.0",
            "tie_rate": "0.0",
            "right_win_rate": "0.0",
            "significant": "True",
        }
    ]
    with (statistics / "posthoc_pairwise.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(posthoc_rows[0]))
        writer.writeheader()
        writer.writerows(posthoc_rows)
    scores = load_scores(statistics)
    output = tmp_path / "axis_scores_main"
    write_scores(dataset="fixture", evaluation_set="main", scores=scores, output=output)
    text = output.with_suffix(".txt").read_text(encoding="utf-8")
    payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert "BASiS-DPO: mean=8.000" in text
    assert "p_holm=0.0008, stars=***" in text
    assert payload["axes"][0]["models"]["Gold-only DPO"]["mean"] == 7.0
