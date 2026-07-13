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
    CONFIGS,
    PREPROCESS_FILES,
    WILDCHAT_FILES,
    reuse_files,
)
from tools.score_dialogue_with_transition_bayes_model import (
    build_transition_scoring_instructions,
    score_single_record,
)
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
    for stage in ("preprocess", "extract_wildchat"):
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


def test_reuse_copies_only_allowlisted_data_and_rejects_config_mismatch(tmp_path: Path):
    project = tmp_path / "project"
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_reuse_source(source, project)
    reuse_files(source, target, mode="preprocess", seed=42, project_root=project)
    reuse_files(source, target, mode="wildchat", seed=42, project_root=project)
    assert (target / PREPROCESS_FILES[1]).exists()
    assert (target / WILDCHAT_FILES[0]).exists()
    assert not (target / "scoring/wildchat_scored.jsonl").exists()
    assert not (target / "basis_model/mathdial_transition_compat.json").exists()
    assert not (target / "stage_state/preprocess_SUCCESS.json").exists()
    (project / CONFIGS[0]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="config hash"):
        reuse_files(source, tmp_path / "other", mode="preprocess", seed=42, project_root=project)
