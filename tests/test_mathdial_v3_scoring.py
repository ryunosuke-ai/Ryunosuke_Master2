from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.transition_bayes_model import parse_transition_bayes_model
from tools.analyze_mathdial_corpus_transition_bayes import (
    evaluate_emission_quality,
    mock_model,
)
from tools.extract_high_posterior_dialogues import (
    derive_selection_label_diagnostics,
    derive_selection_labels_from_model,
)
from tools.reuse_mathdial_pipeline_data import (
    BASIS_FILES,
    CONFIGS,
    PREPROCESS_FILES,
    WILDCHAT_FILES,
    reuse_files,
)
from tools.reuse_transition_scoring import validate_and_reuse_scoring
from tools.prioritize_tutoring_candidates import (
    opportunity_score,
    prioritize_jsonl,
)
from tools.measure_basis_selection_pool import measure_pool
from tools.score_dialogue_with_transition_bayes_model import (
    build_transition_scoring_instructions,
    is_retryable_fallback,
    prepare_retryable_fallback_repair,
    read_new_records_batch,
    score_single_record,
)
from tools.validate_scoring_fallbacks import summarize_fallbacks
from tools.validate_mathdial_scoring_pilot import summarize_pilot
from tools.translate_and_generate_dpo import score_japanese_response


class SequenceGenerator:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.instructions: list[str] = []

    def generate(self, **kwargs):
        self.instructions.append(kwargs["instructions"])
        return self.outputs.pop(0)


def _record() -> dict[str, object]:
    return {
        "conversation_id": "c1",
        "turn_index": 1,
        "prompt": "User: 3 + 4 = 8だと思います。",
        "response": "どの2つの数を足しましたか。",
    }


def test_legacy_scoring_prompt_is_unchanged_by_default():
    model = parse_transition_bayes_model(mock_model())
    assert build_transition_scoring_instructions(model) == build_transition_scoring_instructions(
        model, scoring_preset="legacy"
    )
    assert "会話状態:" in build_transition_scoring_instructions(model)


def test_mathdial_scoring_prompt_exposes_only_observations():
    model = parse_transition_bayes_model(mock_model())
    prompt = build_transition_scoring_instructions(
        model, scoring_preset="mathdial_tutoring"
    )
    assert all(observation in prompt for observation in model.observations)
    assert all(state not in prompt for state in model.states)
    assert model.dataset_hypothesis not in prompt
    assert "助言や説明へ逸れる" not in prompt
    assert "診断後に必要な説明" in prompt


def test_mathdial_invalid_state_label_is_semantically_retried():
    model = parse_transition_bayes_model(mock_model())
    generator = SequenceGenerator(
        [
            json.dumps({"observation": "stalled_misalignment", "score": 0.8, "reason": "x"}),
            json.dumps({"observation": "elicit_reasoning", "score": 0.9, "reason": "再判定"}),
        ]
    )
    result = score_single_record(
        _record(),
        bayes_model=model,
        generator=generator,
        model="terra",
        max_output_tokens=512,
        instructions=build_transition_scoring_instructions(
            model, scoring_preset="mathdial_tutoring"
        ),
        prior_distribution=None,
        progress_label="[test]",
        fallback_on_errors=True,
        scoring_preset="mathdial_tutoring",
        invalid_observation_retries=2,
    )
    assert result["observation"] == "elicit_reasoning"
    assert result["llm_retry"] == "invalid_observation_retry"
    assert "stalled_misalignment" not in generator.instructions[0]
    assert "stalled_misalignment" not in generator.instructions[1]
    assert "許可観測" in generator.instructions[1]


def test_mathdial_falls_back_only_after_semantic_retry_exhaustion():
    model = parse_transition_bayes_model(mock_model())
    invalid = json.dumps({"observation": "stalled_misalignment", "score": 0.8})
    result = score_single_record(
        _record(),
        bayes_model=model,
        generator=SequenceGenerator([invalid, invalid, invalid]),
        model="terra",
        max_output_tokens=512,
        instructions=build_transition_scoring_instructions(
            model, scoring_preset="mathdial_tutoring"
        ),
        prior_distribution=None,
        progress_label="[test]",
        fallback_on_errors=True,
        scoring_preset="mathdial_tutoring",
        invalid_observation_retries=2,
    )
    assert result["llm_error_kind"] == "invalid_observation"
    assert result["observation"] in model.observations


def test_retryable_fallback_repair_removes_whole_conversation(tmp_path: Path):
    records = [
        {"conversation_id": "c1", "turn_index": 1},
        {"conversation_id": "c1", "turn_index": 3},
        {"conversation_id": "c2", "turn_index": 1},
    ]
    existing = [
        {**records[0], "observation": "a"},
        {
            **records[1],
            "observation": "b",
            "llm_error": "RateLimitError: 429 rate limit exceeded",
            "llm_error_kind": "api_or_json",
        },
        {**records[2], "observation": "a"},
    ]
    output = tmp_path / "scored.jsonl"
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in existing), encoding="utf-8"
    )
    retained, conversations = prepare_retryable_fallback_repair(
        records, existing, output
    )
    assert conversations == {"c1"}
    assert [row["conversation_id"] for row in retained] == ["c2"]
    assert [json.loads(line)["conversation_id"] for line in output.open()] == ["c2"]
    assert is_retryable_fallback(existing[1])
    assert not is_retryable_fallback(
        {"llm_error": "content_filter", "llm_error_kind": "content_filter"}
    )


def test_adaptive_batch_skips_scored_conversations(tmp_path: Path):
    records = [
        {
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "prompt": "p",
            "response": "r",
        }
        for conversation_id, size in (("c1", 2), ("c2", 3), ("c3", 2))
        for turn_index in range(size)
    ]
    path = tmp_path / "prioritized.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    existing = [dict(row) for row in records if row["conversation_id"] == "c1"]
    selected = read_new_records_batch(path, existing, max_new_records=3)
    assert {row["conversation_id"] for row in selected} == {"c2"}
    assert len(selected) == 3


def test_priority_uses_only_user_side_and_streams_by_conversation(tmp_path: Path):
    base = {
        "turn_index": 1,
        "history": [{"role": "user", "text": "Teach me a topic."}],
        "next_user_turn": "Continue.",
        "metadata": {},
    }
    plain = {**base, "conversation_id": "plain", "response": "excellent tutor"}
    confused = {
        **base,
        "conversation_id": "confused",
        "history": [
            {
                "role": "user",
                "text": "I tried this equation but my answer is wrong and I am confused?",
            }
        ],
        "response": "unrelated assistant text",
    }
    changed_response = {**confused, "response": "a completely different response"}
    assert opportunity_score(confused) == opportunity_score(changed_response)
    source = tmp_path / "candidates.jsonl"
    source.write_text(
        json.dumps(plain) + "\n" + json.dumps(confused) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prioritized.jsonl"
    report_path = tmp_path / "report.json"
    report = prioritize_jsonl(source, output, report_path=report_path, seed=42)
    rows = [json.loads(line) for line in output.open()]
    assert rows[0]["conversation_id"] == "confused"
    assert report["assistant_response_used_for_priority"] is False
    assert rows[0]["metadata"]["candidate_priority"]["uses_assistant_response"] is False

    original_mtime = output.stat().st_mtime_ns
    second = prioritize_jsonl(source, output, report_path=report_path, seed=42)
    assert second == report
    assert output.stat().st_mtime_ns == original_mtime


def test_selection_pool_measurement_matches_mathdial_selection_conditions(
    tmp_path: Path,
):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(mock_model()), encoding="utf-8")
    rows = []
    for index in range(5):
        rows.append(
            {
                "conversation_id": "same-conversation",
                "turn_index": index,
                "observation": "elicit_reasoning",
                "most_likely_state": "active_diagnosis",
                "posterior": 0.8,
            }
        )
    rows.append(
        {
            "conversation_id": "off-style",
            "turn_index": 0,
            "observation": "generic_repetition",
            "most_likely_state": "stalled_misalignment",
            "posterior": 0.9,
        }
    )
    report = measure_pool(
        rows,
        model_path=model_path,
        method="state_specific_margin",
        margin=0.05,
    )
    assert report["eligible_records"] == 3
    assert report["eligible_conversations"] == 1


def test_clean_selection_pool_excludes_whole_fallback_conversation(tmp_path: Path):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(mock_model()), encoding="utf-8")
    rows = [
        {
            "conversation_id": "dirty",
            "turn_index": 0,
            "observation": "generic_repetition",
            "most_likely_state": "stalled_misalignment",
            "posterior": 0.2,
            "llm_error": "RateLimitError: 429",
        },
        {
            "conversation_id": "dirty",
            "turn_index": 1,
            "observation": "elicit_reasoning",
            "most_likely_state": "active_diagnosis",
            "posterior": 0.8,
        },
        {
            "conversation_id": "clean",
            "turn_index": 0,
            "observation": "elicit_reasoning",
            "most_likely_state": "active_diagnosis",
            "posterior": 0.8,
        },
    ]
    report = measure_pool(
        rows,
        model_path=model_path,
        method="state_specific_margin",
        margin=0.05,
        exclude_fallback_conversations=True,
    )
    assert report["eligible_records_before_fallback_exclusion"] == 2
    assert report["eligible_records"] == 1
    assert report["excluded_eligible_records"] == 1
    assert report["fallback_conversations"] == 1


def test_selection_pool_counts_only_complete_histories_within_length_limit(
    tmp_path: Path,
):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(mock_model()), encoding="utf-8")
    rows = [
        {
            "conversation_id": "short",
            "turn_index": 0,
            "prompt": "question",
            "response": "hint",
            "observation": "elicit_reasoning",
            "most_likely_state": "active_diagnosis",
            "posterior": 0.8,
        },
        {
            "conversation_id": "long",
            "turn_index": 0,
            "prompt": "x" * 200,
            "response": "hint",
            "observation": "elicit_reasoning",
            "most_likely_state": "active_diagnosis",
            "posterior": 0.8,
        },
    ]
    report = measure_pool(
        rows,
        model_path=model_path,
        method="state_specific_margin",
        margin=0.05,
        max_source_characters=100,
    )
    assert report["eligible_records"] == 1
    assert report["records_over_length_limit"] == 1
    assert report["length_filter_policy"] == (
        "exclude_whole_sample_without_truncating_history"
    )


def test_full_fallback_gate_warns_before_fatal():
    records = [{"conversation_id": f"c{i}"} for i in range(100)]
    for index in range(3):
        records[index]["llm_error"] = "RateLimitError: 429"
    warning = summarize_fallbacks(records, warning_rate=0.01, fatal_rate=0.05)
    assert warning["passed"]
    assert warning["warning"]
    assert warning["fallback_reason_distribution"] == {"rate_limit": 3}
    fatal = summarize_fallbacks(records, warning_rate=0.01, fatal_rate=0.02)
    assert not fatal["passed"]


def test_mathdial_dpo_rescoring_uses_mathdial_preset_and_semantic_retry():
    model = parse_transition_bayes_model(mock_model())
    generator = SequenceGenerator(
        [
            json.dumps({"observation": "guided_scaffolding", "score": 0.7}),
            json.dumps({"observation": "focused_hint", "score": 0.9, "reason": "焦点化"}),
        ]
    )
    result = score_japanese_response(
        record={
            "conversation_id": "c1",
            "turn_index": 1,
            "prompt": "User: 途中式が分かりません。",
        },
        response="まずどの式を使うか考えてみましょう。",
        bayes_model=model,
        generator=generator,
        score_model="terra",
        max_output_tokens=512,
        style_preset="mathdial_tutoring",
    )
    assert result["observation"] == "focused_hint"
    assert result["llm_retry"] == "invalid_observation_retry"
    assert all(state not in generator.instructions[0] for state in model.states)


def test_emission_gate_rejects_v2_like_stalled_state():
    payload = mock_model()
    payload["emission_likelihoods"]["stalled_misalignment"] = dict(
        payload["emission_likelihoods"]["verified_understanding"]
    )
    report = evaluate_emission_quality(payload)
    assert not report["passed"]
    assert not report["negative_state_discriminators"]["stalled_misalignment"]["passed"]
    assert len(report["negative_dominant_observations"]) < 2


def test_emission_gate_accepts_distinct_off_style_observations():
    report = evaluate_emission_quality(mock_model())
    assert report["passed"]
    assert len(report["negative_dominant_observations"]) >= 2


def test_state_specific_selection_prefers_diagnosed_explanation():
    model = parse_transition_bayes_model(mock_model())
    details = derive_selection_label_diagnostics(
        model, method="state_specific_margin", minimum_margin=0.05
    )
    assert details["observations"]["diagnosed_explanation"]["classification"] == "preferred"
    assert "diagnosed_explanation" in details["labels"]["prefer_observations"]


def test_mean_difference_default_remains_legacy_compatible():
    model = parse_transition_bayes_model(mock_model())
    assert derive_selection_labels_from_model(model) == derive_selection_labels_from_model(
        model, method="mean_difference"
    )


def test_pilot_gate_rejects_excess_fallback_and_accepts_diverse_valid_rows():
    allowed = {"a", "b"}
    valid = [
        {"observation": "a" if index % 2 else "b"}
        for index in range(200)
    ]
    assert summarize_pilot(
        valid,
        allowed_observations=allowed,
        required_records=200,
        max_fallback_rate=0.01,
        max_invalid_rate=0.01,
        min_observations=2,
    )["passed"]
    invalid = list(valid)
    for index in range(3):
        invalid[index] = {
            "observation": "b",
            "llm_error": "unknown",
            "llm_error_kind": "invalid_observation",
        }
    assert not summarize_pilot(
        invalid,
        allowed_observations=allowed,
        required_records=200,
        max_fallback_rate=0.01,
        max_invalid_rate=0.01,
        min_observations=2,
    )["passed"]


def _write_reuse_source(source: Path, project: Path, *, seed: int = 42) -> None:
    config_hashes = {}
    for relative in CONFIGS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        config_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "stage_state").mkdir(parents=True)
    (source / "run_metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "experiment_fingerprint": "source-fingerprint",
                "configs": config_hashes,
            }
        ),
        encoding="utf-8",
    )
    for stage in ("preprocess", "build_basis", "extract_wildchat"):
        (source / "stage_state" / f"{stage}_SUCCESS.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "seed": seed,
                    "experiment_fingerprint": "source-fingerprint",
                }
            ),
            encoding="utf-8",
        )
    for relative in (*PREPROCESS_FILES, *WILDCHAT_FILES):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix in {".json", ".jsonl"} else "report\n", encoding="utf-8")
    forbidden = source / "scoring/wildchat_scored.jsonl"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("{}\n", encoding="utf-8")
    model = source / "basis_model/mathdial_transition_compat.json"
    model.parent.mkdir(parents=True)
    model.write_text("{}\n", encoding="utf-8")


def _write_valid_basis_source(source: Path) -> None:
    basis = source / "basis_model"
    basis.mkdir(parents=True, exist_ok=True)
    payload = mock_model()
    model_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fine = basis / "mathdial_transition_bayes_model.json"
    compat = basis / "mathdial_transition_compat.json"
    fine.write_text(model_text, encoding="utf-8")
    compat.write_text(model_text, encoding="utf-8")
    analysis = basis / "mathdial_analysis_corpus.jsonl"
    analysis.write_text('{"source_split":"train"}\n', encoding="utf-8")
    conversation = source / "mathdial/data/mathdial_conversations.jsonl"
    (basis / "mathdial_analysis_corpus.manifest.json").write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(conversation.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (basis / "mathdial_transition_bayes_model.manifest.json").write_text(
        json.dumps({"output_sha256": hashlib.sha256(fine.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    quality = evaluate_emission_quality(payload)
    (basis / "mathdial_model_quality.json").write_text(
        json.dumps(quality), encoding="utf-8"
    )
    (basis / "mathdial_analysis_input.txt").write_text("input\n", encoding="utf-8")
    (basis / "mathdial_analysis_prompt.txt").write_text("prompt\n", encoding="utf-8")


def test_reuse_copies_only_allowlisted_data_and_rejects_config_mismatch(tmp_path: Path):
    project = tmp_path / "project"
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_reuse_source(source, project)
    reuse_files(source, target, mode="preprocess", seed=42, project_root=project)
    reuse_files(source, target, mode="wildchat", seed=42, project_root=project)
    assert (target / "wildchat/general_tutoring_candidates.jsonl").is_symlink()
    assert (target / PREPROCESS_FILES[1]).exists()
    assert (target / WILDCHAT_FILES[0]).exists()
    assert not (target / "scoring/wildchat_scored.jsonl").exists()
    assert not (target / "basis_model/mathdial_transition_compat.json").exists()
    assert not (target / "stage_state/preprocess_SUCCESS.json").exists()
    (project / CONFIGS[0]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="config hash"):
        reuse_files(source, tmp_path / "other", mode="preprocess", seed=42, project_root=project)


def test_reuse_basis_revalidates_quality_without_copying_scoring(tmp_path: Path):
    project = tmp_path / "project"
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_reuse_source(source, project)
    _write_valid_basis_source(source)
    reuse_files(source, target, mode="basis", seed=42, project_root=project)
    assert all((target / relative).exists() for relative in BASIS_FILES)
    assert not (target / "scoring/wildchat_scored.jsonl").exists()
    assert not (target / "stage_state/build_basis_SUCCESS.json").exists()
    manifest = json.loads((target / "reuse_manifest.json").read_text(encoding="utf-8"))
    assert manifest["modes"]["basis"]["emission_quality"]["passed"]


def test_reuse_complete_scoring_validates_candidates_and_pilot(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    candidates = [
        {
            "sample_id": f"s{i}",
            "conversation_id": f"c{i}",
            "turn_index": 1,
            "prompt": f"p{i}",
            "response": f"r{i}",
        }
        for i in range(3)
    ]
    scored = [
        {
            **row,
            "observation": "a",
            "state_posteriors": {"positive": 0.8, "negative": 0.2},
        }
        for row in candidates
    ]
    for root in (source, target):
        (root / "wildchat").mkdir(parents=True)
        (root / "scoring").mkdir(parents=True)
        (root / "basis_model").mkdir(parents=True)
        (root / "wildchat/general_tutoring_candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
        )
        (root / "scoring/prioritized_candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
        )
        (root / "basis_model/mathdial_transition_compat.json").write_text(
            "same-model", encoding="utf-8"
        )
        (root / "run_metadata.json").write_text(
            json.dumps(
                {
                    "models": {"scoring": "terra"},
                    "scoring": {
                        "preset": "mathdial_tutoring",
                        "preset_version": "mathdial_v3",
                    },
                }
            ),
            encoding="utf-8",
        )
    (source / "scoring/wildchat_scored_raw.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scored), encoding="utf-8"
    )
    (source / "scoring/pilot_diagnostics.json").write_text(
        json.dumps({"passed": True, "records": 3}), encoding="utf-8"
    )
    with (target / "scoring/prioritized_candidates.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(
                {
                    "sample_id": "s-extra",
                    "conversation_id": "c-extra",
                    "turn_index": 1,
                    "prompt": "extra prompt",
                    "response": "extra response",
                }
            )
            + "\n"
        )
    report = validate_and_reuse_scoring(
        source_root=source,
        target_root=target,
        expected_records=None,
    )
    assert report["records"] == 3
    assert (target / "scoring/wildchat_scored_raw.jsonl").exists()
    assert (target / "scoring/pilot_diagnostics.json").exists()
