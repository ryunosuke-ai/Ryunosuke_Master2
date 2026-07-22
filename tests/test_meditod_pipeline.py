from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.dpo_prompting import (
    MEDITOD_DPO_PROMPT_TEMPLATE_VERSION,
    build_meditod_dpo_prompt,
)
from tools.analyze_meditod_corpus_transition_bayes import (
    DEFAULT_MAX_INPUT_CHARS,
    build_meditod_corpus_text,
    evaluate_model_quality,
    mock_model,
)
from tools.meditod_annotation_metrics import compute
from tools.meditod_dataset import (
    SourceDialogue,
    content_hash,
    convert_raw_group,
    load_public_raw,
    load_yaml,
    prepare_public_raw,
    summarize,
)
from tools.meditod_evaluation import (
    evaluation_translation_fidelity_errors,
    select_eval_prompts,
)
from tools.score_dialogue_with_transition_bayes_model import (
    build_meditod_scoring_instructions,
)
from tools.translate_and_generate_dpo import meditod_translation_fidelity_errors
from tools.wildchat_health import extract_candidates, health_domain_flags
from tools.wildchat_health import has_explicit_unsafe_medical_advice


FIXTURES = Path("tests/fixtures")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_raw_parser_role_mapping_control_and_history():
    config = load_yaml(FIXTURES / "meditod_public_raw_config.yaml")
    conversations, samples, report = prepare_public_raw(
        FIXTURES / "meditod_dialogs.json",
        FIXTURES / "meditod_annotations.json",
        config=config,
        seed=42,
    )
    assert len(conversations) == 5
    assert report["source"]["control_turns_removed_from_text"] == 5
    assert report["leakage_check"]["status"] == "passed"
    assert all(turn["role"] in {"user", "assistant"} for row in conversations for turn in row["turns"])
    assert all(turn["metadata"]["source_speaker"] == ("doctor" if turn["role"] == "assistant" else "patient") for row in conversations for turn in row["turns"])
    sample = next(row for row in samples if row["history"] and row["next_user_turn"])
    assert sample["history"][-1]["role"] == "user"
    assert sample["metadata"]["response_intents"]
    assert sample["metadata"]["after_state_observed"] is True


def test_duplicate_dialogues_merge_and_keep_annotation_variants():
    first = load_public_raw(FIXTURES / "meditod_dialogs.json", FIXTURES / "meditod_annotations.json")[0]
    annotations = json.loads(json.dumps(first.annotations))
    annotations[0][0]["intent"] = "inquire"
    second = SourceDialogue("duplicate", "r1_duplicate", json.loads(json.dumps(first.utterances)), annotations)
    digest = content_hash(first.utterances)
    conversation = convert_raw_group(digest, [first, second], split="train", mode="public_raw", ood_prefix="r3_")
    assert conversation["metadata"]["duplicate_records_merged"] == 1
    assert conversation["metadata"]["annotation_variant_count"] == 2
    assert len(conversation["turns"][0]["metadata"]["annotation_variants"]) == 2
    report = summarize([first, second], [conversation])
    assert report["normalized"]["annotation_disagreement_turns"] >= 1


def test_meditod_bayes_quality_and_scoring_prompt_hide_states():
    payload = mock_model()
    report = evaluate_model_quality(payload, emission_margin=0.10, min_negative=2)
    assert report["passed"] is True
    from core.transition_bayes_model import parse_transition_bayes_model

    model = parse_transition_bayes_model(payload)
    prompt = build_meditod_scoring_instructions(model)
    assert all(observation in prompt for observation in model.observations)
    assert all(state not in prompt for state in model.states)
    assert "observation" in prompt
    assert "診断や助言を急いでいないことを粗抽出条件" not in prompt


def test_meditod_analysis_limit_keeps_complete_public_raw_sample():
    """公開raw版24診療の実測サイズを切らずに扱える余裕を維持する。"""
    records = [
        {
            "conversation_id": "meditod_train_000",
            "source_split": "train",
            "dialog": [
                {
                    "turn_index": 0,
                    "speaker": "doctor",
                    "text": "x" * 677_281,
                    "annotation_variants": [],
                }
            ],
        }
    ]
    corpus = build_meditod_corpus_text(
        records,
        {},
        max_chars=DEFAULT_MAX_INPUT_CHARS,
    )
    assert len(corpus) > 677_281
    assert DEFAULT_MAX_INPUT_CHARS == 800_000
    with pytest.raises(ValueError, match="max-input-chars"):
        build_meditod_corpus_text(records, {}, max_chars=400_000)


def test_wildchat_health_filter_is_domain_and_multiturn_only():
    config = load_yaml("configs/datasets/wildchat_health.yaml")
    rows = [json.loads(line) for line in (FIXTURES / "wildchat_health.jsonl").read_text().splitlines() if line]
    general, respiratory, stats = extract_candidates(rows, config, progress_every=0, checkpoint_every=0)
    assert len(general) == 2
    assert len(respiratory) == 1
    assert stats["general_candidate_records"] >= 2
    flags = health_domain_flags(general[0], config)
    assert flags["health"] and flags["followup_information"]
    all_text = " ".join(turn["text"].lower() for turn in general[0]["turns"])
    assert "probing" not in all_text and "scaffolding" not in all_text


def test_meditod_prompt_and_translation_fidelity():
    prompt = build_meditod_dpo_prompt(
        history_turns=[
            {"speaker": "User", "text": "3日前から咳があります。"},
            {"speaker": "AI", "text": "熱はありますか。"},
            {"speaker": "User", "text": "熱はありません。"},
        ]
    )
    assert prompt.startswith("以下の医療相談の会話の文脈に沿って")
    assert prompt.endswith("AI:")
    assert "診断を急が" not in prompt and "病歴聴取" not in prompt
    assert MEDITOD_DPO_PROMPT_TEMPLATE_VERSION
    source = {"prompt": "I have no fever for 3 days and take metformin 500 mg.", "response": "Any cough?", "metadata": {"protected_medical_terms": ["metformin"]}}
    good = {"translated_prompt": "3日間、発熱はなく、metformin 500 mgを服用しています。", "translated_chosen": "咳はありますか。", "rejected_candidates": ["発熱があります。"]}
    bad = {**good, "translated_prompt": "発熱があり、薬を飲んでいます。"}
    assert not meditod_translation_fidelity_errors(source, good)
    assert meditod_translation_fidelity_errors(source, bad)
    unsafe = {**good, "rejected_candidates": ["metformin 1000 mgを服用してください。"]}
    assert "explicit_unsafe_medical_advice" in meditod_translation_fidelity_errors(source, unsafe)
    assert has_explicit_unsafe_medical_advice("Take 500 mg of this medicine.")


def test_random_dpo_can_report_shortfall_to_adaptive_pipeline(tmp_path: Path):
    """Random候補不足を保存して、pipelineが追加scoringできる。"""
    source = tmp_path / "source.jsonl"
    output = tmp_path / "random.jsonl"
    source.write_text(
        json.dumps(
            {
                "conversation_id": "c1",
                "turn_index": 1,
                "prompt": "User: I have a cough.",
                "response": "How long have you had it?",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.build_random_dailydialog_dpo",
            "--input",
            str(source),
            "--output",
            str(output),
            "--daily-output",
            str(output),
            "--target-records",
            "2",
            "--max-source-records",
            "0",
            "--allow-target-shortfall",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == ""


def test_eval_selection_and_annotation_metrics_are_cluster_aware_inputs():
    config = load_yaml(FIXTURES / "meditod_public_raw_config.yaml")
    _, samples, _ = prepare_public_raw(FIXTURES / "meditod_dialogs.json", FIXTURES / "meditod_annotations.json", config=config, seed=42)
    selected, manifest = select_eval_prompts(samples, count=2, seed=42, ood=False, max_per_consultation=6)
    assert len(selected) == 2
    assert manifest["selection_uses_model_outputs"] is False
    assert manifest["selection_uses_oracle_scores"] is False
    metric_rows = compute([
        {"sample_id": "s1", "conversation_id": "c1", "selection_stratum": "symptom_attributes", "history_ja": [{"role": "assistant", "text": "熱はありますか。"}, {"role": "user", "text": "ありません。"}], "base_response": "熱はありますか。", "basis_response": "咳はいつ始まりましたか。", "random_dpo_response": "受診してください。", "source_response_slots": ["symptom"]}
    ])
    assert len(metric_rows) == 3
    base = next(row for row in metric_rows if row["model_name"] == "base")
    basis = next(row for row in metric_rows if row["model_name"] == "basis")
    assert base["duplicate_question_similarity"] > basis["duplicate_question_similarity"]


def test_eval_translation_fidelity_catches_medical_information_loss():
    source = {
        "history_en": [
            {"role": "user", "text": "I have had no fever for 3 days."},
            {"role": "assistant", "text": "Do you take metformin 500 mg?"},
        ],
        "reference_response_en": "Has the cough become worse?",
        "next_patient_turn_en": "The cough is worse at night.",
    }
    good = {
        "history_ja": [
            {"role": "user", "text": "3日間、発熱はありません。"},
            {"role": "assistant", "text": "metformin 500 mgを服用していますか。"},
        ],
        "reference_response_ja": "咳は悪化しましたか。",
        "next_patient_turn_ja": "咳は夜に悪化します。",
    }
    bad = {
        **good,
        "history_ja": [
            {"role": "user", "text": "発熱があります。"},
            {"role": "assistant", "text": "薬を服用していますか。"},
        ],
    }
    assert evaluation_translation_fidelity_errors(source, good) == {}
    errors = evaluation_translation_fidelity_errors(source, bad)
    assert {"numbers", "negation", "medications"}.issubset(errors)
