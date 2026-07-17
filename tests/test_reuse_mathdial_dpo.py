"""MathDial DPO採択済み結果の条件付き再利用テスト。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.dpo_prompting import DPO_PROMPT_TEMPLATE_VERSION
from tools.reuse_mathdial_dpo import reuse_accepted_records
from tools.translate_and_generate_dpo import PROMPT_TEMPLATE_VERSION, bayes_model_version


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def _model(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "fixture",
                "model_type": "transition_bayes_network",
                "states": ["positive", "negative"],
                "positive_states": ["positive"],
                "negative_states": ["negative"],
                "observations": ["guided", "off_style"],
                "initial_state_prior": {"positive": 0.5, "negative": 0.5},
                "transition_likelihoods": {
                    "positive": {"positive": 0.8, "negative": 0.2},
                    "negative": {"positive": 0.2, "negative": 0.8},
                },
                "emission_likelihoods": {
                    "positive": {"guided": 0.8, "off_style": 0.2},
                    "negative": {"guided": 0.2, "off_style": 0.8},
                },
                "state_descriptions": {"positive": "target", "negative": "other"},
                "observation_descriptions": {"guided": "guide", "off_style": "other"},
                "dataset_hypothesis": "fixture",
            }
        ),
        encoding="utf-8",
    )


def _accepted(
    selection: dict,
    model_version: str,
    *,
    scoring_model: str = "terra",
    candidates: int = 8,
) -> dict:
    prompt = selection["prompt"]
    return {
        "prompt": "日本語prompt",
        "chosen": "chosen",
        "rejected": "rejected",
        "score_chosen": 0.9,
        "score_rejected": 0.3,
        "score_gap": 0.6,
        "source_dialogue_id": selection["conversation_id"],
        "turn_index": selection["turn_index"],
        "source_prompt_en": prompt,
        "source_chosen_en": selection["response"],
        "model_used_for_translation": "terra",
        "model_used_for_rejected_generation": "terra",
        "model_used_for_scoring": scoring_model,
        "bayesian_model_version": model_version,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "dpo_prompt_template_version": DPO_PROMPT_TEMPLATE_VERSION,
        "metadata": {
            "source_prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "translated_prompt_hash": "same",
            "rejected_prompt_hash": "same",
            "style_preset": "mathdial_tutoring",
            "rejected_candidates": candidates,
            "seed": 42,
        },
    }


def test_reuses_only_matching_accepted_and_never_copies_skips(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    _model(model)
    version = bayes_model_version(model)
    valid = {"conversation_id": "c1", "turn_index": 1, "prompt": "short", "response": "answer"}
    long = {"conversation_id": "c2", "turn_index": 1, "prompt": "x" * 100, "response": "answer"}
    mismatch = {"conversation_id": "c3", "turn_index": 1, "prompt": "short3", "response": "answer3"}
    selection = tmp_path / "selection.jsonl"
    source = tmp_path / "old" / "basis_selected_ja.jsonl"
    output = tmp_path / "new" / "basis_selected_ja.jsonl"
    manifest = tmp_path / "new" / "manifest.json"
    _write_jsonl(selection, [valid, long, mismatch])
    _write_jsonl(
        source,
        [
            _accepted(valid, version),
            _accepted(long, version),
            _accepted(mismatch, version, scoring_model="other"),
        ],
    )
    _write_jsonl(tmp_path / "old" / "basis_selected_ja_skipped.jsonl", [{"irrelevant": True}])

    report = reuse_accepted_records(
        source_output=source,
        current_selection=selection,
        bayes_model=model,
        output=output,
        manifest=manifest,
        generation_model="terra",
        scoring_model="terra",
        style_preset="mathdial_tutoring",
        candidates=8,
        seed=42,
        max_source_characters=50,
        min_score_gap=0.2,
        min_chosen_posterior=0.7,
        max_rejected_posterior=0.55,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["source_dialogue_id"], row["turn_index"]) for row in rows] == [("c1", 1)]
    assert report["inherited_records"] == 1
    assert report["skipped_records_reused"] == 0
    assert report["rejected_reasons"] == {
        "scoring_model_mismatch": 1,
        "source_too_long": 1,
    }


def test_merges_with_current_output_without_duplicates(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    _model(model)
    selection_row = {"conversation_id": "c1", "turn_index": 1, "prompt": "p", "response": "r"}
    accepted = _accepted(selection_row, bayes_model_version(model))
    source = tmp_path / "source.jsonl"
    selection = tmp_path / "selection.jsonl"
    output = tmp_path / "output.jsonl"
    _write_jsonl(source, [accepted])
    _write_jsonl(selection, [selection_row])
    _write_jsonl(output, [accepted])

    report = reuse_accepted_records(
        source_output=source,
        current_selection=selection,
        bayes_model=model,
        output=output,
        manifest=tmp_path / "manifest.json",
        generation_model="terra",
        scoring_model="terra",
        style_preset="mathdial_tutoring",
        candidates=8,
        seed=42,
        max_source_characters=16000,
        min_score_gap=0.2,
        min_chosen_posterior=0.7,
        max_rejected_posterior=0.55,
    )

    assert report["inherited_records"] == 0
    assert report["output_records"] == 1


def test_mixed_candidate_continuation_reuses_accepted_and_threshold_skips(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.json"
    _model(model)
    version = bayes_model_version(model)
    accepted_selection = {
        "conversation_id": "accepted",
        "turn_index": 1,
        "prompt": "prompt accepted",
        "response": "response accepted",
    }
    skipped_selection = {
        "conversation_id": "skipped",
        "turn_index": 1,
        "prompt": "prompt skipped",
        "response": "response skipped",
    }
    error_selection = {
        "conversation_id": "error",
        "turn_index": 1,
        "prompt": "prompt error",
        "response": "response error",
    }
    selection = tmp_path / "selection.jsonl"
    source = tmp_path / "old" / "basis_selected_ja.jsonl"
    source_skipped = tmp_path / "old" / "basis_selected_ja_skipped.jsonl"
    output = tmp_path / "new" / "basis_selected_ja.jsonl"
    skipped_output = tmp_path / "new" / "basis_selected_ja_skipped.jsonl"
    _write_jsonl(selection, [accepted_selection, skipped_selection, error_selection])
    _write_jsonl(source, [_accepted(accepted_selection, version, candidates=8)])
    low_chosen = _accepted(skipped_selection, version, candidates=8)
    low_chosen.update(
        {
            "skip_reason": "low_chosen",
            "score_chosen": 0.4,
            "score_rejected": 0.3,
            "score_gap": 0.1,
        }
    )
    sample_error = {
        "source_dialogue_id": "error",
        "turn_index": 1,
        "skip_reason": "sample_error",
    }
    _write_jsonl(source_skipped, [low_chosen, sample_error])

    report = reuse_accepted_records(
        source_output=source,
        source_skipped=source_skipped,
        current_selection=selection,
        bayes_model=model,
        output=output,
        skipped_output=skipped_output,
        manifest=tmp_path / "manifest.json",
        generation_model="terra",
        scoring_model="terra",
        style_preset="mathdial_tutoring",
        candidates=4,
        allowed_source_candidates={8},
        reuse_threshold_skips=True,
        seed=42,
        max_source_characters=16000,
        min_score_gap=0.2,
        min_chosen_posterior=0.7,
        max_rejected_posterior=0.55,
    )

    assert report["inherited_records"] == 1
    assert report["skipped_records_reused"] == 1
    assert report["skipped_rejected_reasons"] == {"non_threshold_skip": 1}
    assert len(output.read_text().splitlines()) == 1
    assert len(skipped_output.read_text().splitlines()) == 1
