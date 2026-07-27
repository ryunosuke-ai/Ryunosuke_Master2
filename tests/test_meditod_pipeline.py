from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
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
from tools.meditod_available_data_decision import (
    validate_available_data_decision,
)
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
    finalize_translated_prompts,
    select_eval_prompt_candidates,
    select_eval_prompts,
)
from tools.prepare_meditod_personal_pool import audit_resume_records
from tools.prepare_meditod_broad_pool import (
    restore_broad_resume_records,
    verify_broad_pool_artifacts,
)
from tools.promote_meditod_dpo_rescue import rescue_eligible
from tools.score_dialogue_with_transition_bayes_model import (
    build_meditod_scoring_instructions,
)
from tools.translate_and_generate_dpo import (
    MEDITOD_MEDICAL_FIDELITY_VERSION,
    meditod_translation_fidelity_errors,
    missing_meditod_numeric_tokens,
    retry_meditod_translation_for_fidelity,
)
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


def test_wildchat_broad_health_keeps_non_personal_multiturn_task():
    config = load_yaml("configs/datasets/wildchat_health.yaml")
    assert config["require_personal_consultation"] is False
    rows = [
        {
            "language": "English",
            "toxic": False,
            "redacted": False,
            "model": "gpt-4",
            "conversation": [
                {
                    "role": "user",
                    "content": "Please summarize this medical research article.",
                },
                {"role": "assistant", "content": "Which section should I summarize?"},
                {"role": "user", "content": "Start with the diagnosis section."},
                {"role": "assistant", "content": "What level of detail do you need?"},
                {"role": "user", "content": "Also cover the treatment results."},
                {"role": "assistant", "content": "Should I include limitations?"},
                {"role": "user", "content": "Yes, include the clinical limitations too."},
                {"role": "assistant", "content": "I will summarize those sections."},
            ],
        }
    ]
    general, _, _ = extract_candidates(
        rows,
        config,
        progress_every=0,
        checkpoint_every=0,
    )
    assert len(general) == 1
    assert health_domain_flags(general[0], config)["personal"] is False


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
    assert MEDITOD_MEDICAL_FIDELITY_VERSION == "meditod_medical_fidelity.v3"


def test_meditod_fidelity_ignores_citation_numbers_but_keeps_clinical_numbers():
    source = {
        "prompt": "Dopamine is discussed in prior work [144–146] and [4, 149]. I take 500 mg for 3 days.",
        "response": "Do you still take 500 mg?",
        "metadata": {"protected_medical_terms": []},
    }
    good = {
        "translated_prompt": "ドパミンは先行研究で論じられています。500 mgを3日間服用しています。",
        "translated_chosen": "現在も500 mgを服用していますか。",
        "rejected_candidates": ["薬について確認できますか。"],
    }
    assert meditod_translation_fidelity_errors(source, good) == {}
    bad = {**good, "translated_prompt": "薬を3日間服用しています。"}
    assert meditod_translation_fidelity_errors(source, bad)["prompt_numbers"] == ["500"]


def test_meditod_fidelity_ignores_real_citation_series_and_geology_numbers():
    citation_source = (
        "Dopamine is linked to schizophrenia [144–146], receptor changes [147], "
        "D2 affinity [4, 149], and imaging findings [150–178]."
    )
    assert missing_meditod_numeric_tokens(
        citation_source,
        "ドパミンは統合失調症や受容体変化、画像所見と関連します。",
    ) == []
    geology_source = (
        "The extinction happened 252 million years ago. A later model used "
        "720000 observations across 24 sections."
    )
    assert missing_meditod_numeric_tokens(
        geology_source,
        "大量絶滅は非常に古く、後のモデルでは多くの観測を使いました。",
    ) == []
    assert missing_meditod_numeric_tokens(
        "I take 500 mg for my pain.",
        "痛み止めを服用しています。",
    ) == ["500"]
    assert missing_meditod_numeric_tokens(
        "My cough started 3 days ago.",
        "咳が始まりました。",
    ) == ["3"]


def test_health_filter_uses_word_boundaries_and_requires_personal_consultation():
    config = load_yaml("configs/datasets/wildchat_health.yaml")
    false_positive = {
        "turns": [
            {"role": "user", "text": "Write a manga set in Spain."},
            {"role": "assistant", "text": "What genre?"},
            {"role": "user", "text": "A hospital drama."},
        ]
    }
    personal = {
        "turns": [
            {"role": "user", "text": "I have chest pain and a cough."},
            {"role": "assistant", "text": "When did it start?"},
            {"role": "user", "text": "It started 3 days ago and is worse."},
        ]
    }
    assert not health_domain_flags(false_positive, config)["personal"]
    assert health_domain_flags(personal, config)["personal"]


def test_health_filter_rejects_task_drift_after_greeting():
    config = load_yaml("configs/datasets/wildchat_health.yaml")
    writing_task = {
        "turns": [
            {"role": "user", "text": "Hi there!"},
            {"role": "assistant", "text": "Hello."},
            {
                "role": "user",
                "text": (
                    "Please summarize this medical research paper. "
                    "I have included a passage about pain and medication."
                ),
            },
            {"role": "assistant", "text": "Please share it."},
            {"role": "user", "text": "The patient had pain for three days."},
        ]
    }
    delayed_consultation = {
        "turns": [
            {"role": "user", "text": "Are you there?"},
            {"role": "assistant", "text": "Yes."},
            {
                "role": "user",
                "text": "I have chest pain and a cough that started three days ago.",
            },
            {"role": "assistant", "text": "Is the pain worse when breathing?"},
            {"role": "user", "text": "Yes, and I also feel dizzy."},
        ]
    }
    assert not health_domain_flags(writing_task, config)["personal"]
    assert health_domain_flags(delayed_consultation, config)["personal"]


def test_meditod_targeted_fidelity_repair_preserves_rejected_candidates():
    class RepairGenerator:
        def generate(self, **kwargs):
            return json.dumps(
                {
                    "translated_prompt": "500 mgを3日間服用しています。",
                    "translated_chosen": "痛みはまだ続いていますか。",
                },
                ensure_ascii=False,
            )

    source = {
        "prompt": "I have taken 500 mg for 3 days.",
        "response": "Does the pain continue?",
        "metadata": {"protected_medical_terms": []},
    }
    payload = {
        "translated_prompt": "薬を3日間服用しています。",
        "translated_chosen": "痛みはまだ続いていますか。",
        "rejected_candidates": ["すぐに診断できます。"],
    }
    repaired = retry_meditod_translation_for_fidelity(
        source_record=source,
        payload=payload,
        index=0,
        generator=RepairGenerator(),
        instructions="unused",
        model="mock",
        max_output_tokens=1000,
        candidates=4,
        seed=42,
    )
    assert repaired["rejected_candidates"] == payload["rejected_candidates"]
    assert repaired["generation_retry"] == "meditod_medical_fidelity_targeted_retry"
    assert meditod_translation_fidelity_errors(source, repaired) == {}


def test_resume_audit_keeps_domain_records_and_requeues_old_fidelity_errors():
    accepted = [
        {"source_dialogue_id": "health", "turn_index": 1},
        {"source_dialogue_id": "comic", "turn_index": 1},
    ]
    skipped = [
        {
            "source_dialogue_id": "health",
            "turn_index": 3,
            "skip_reason": "sample_error",
            "error_message": "MediTOD翻訳で医療情報が失われました: citation",
        },
        {
            "source_dialogue_id": "health",
            "turn_index": 5,
            "skip_reason": "low_chosen",
        },
    ]
    audit = audit_resume_records(
        accepted=accepted,
        skipped=skipped,
        allowed_ids={"health"},
    )
    assert len(audit["accepted"]) == 1
    assert len(audit["accepted_quarantine"]) == 1
    assert len(audit["fidelity_retry"]) == 1
    assert len(audit["skipped"]) == 1


def test_broad_pool_verification_and_resume_restore():
    config = load_yaml("configs/datasets/wildchat_health.yaml")
    conversation = {
        "conversation_id": "health",
        "turns": [
            {"role": "user", "text": "I have had a cough for 3 days."},
            {"role": "assistant", "text": "Do you take any medicine?"},
            {"role": "user", "text": "No medicine."},
        ],
    }
    source = {
        "conversation_id": "health",
        "turn_index": 1,
        "prompt": "User: I have had a cough for 3 days.\nAI:",
        "response": "Do you take any medicine?",
        "metadata": {
            "personal_health_consultation": True,
            "protected_medical_terms": [],
        },
    }
    manifest = {
        "dataset": config["dataset_name"],
        "revision": config["revision"],
        "stream_shuffle_seed": 42,
        "config": {**config, "require_personal_consultation": False},
        "statistics": {
            "stream_rows": 10,
            "stream_exhausted": 1,
            "general_conversations": 1,
            "general_candidate_records": 1,
        },
    }
    report = verify_broad_pool_artifacts(
        config=config,
        conversations=[conversation],
        candidates=[source],
        manifest=manifest,
        statistics={"stream_exhausted": 1},
        seed=42,
    )
    assert report["broad_candidate_records"] == 1
    assert report["personal_filter_affects_main_eligibility"] is False

    source_hash = __import__("hashlib").sha256(
        source["prompt"].encode("utf-8")
    ).hexdigest()
    accepted = {
        "source_dialogue_id": "health",
        "turn_index": 1,
        "prompt": "医療相談\nAI:",
        "chosen": "3日間の咳なのですね。薬は服用していますか。",
        "rejected": "分かりません。",
        "raw_translated_prompt": "3日間、咳があります。",
        "translated_chosen": "3日間の咳なのですね。薬は服用していますか。",
        "score_chosen": 0.9,
        "score_rejected": 0.4,
        "score_gap": 0.5,
        "model_used_for_translation": "terra",
        "model_used_for_scoring": "terra",
        "bayesian_model_version": "basis-v1",
        "metadata": {
            "source_prompt_hash": source_hash,
            "translated_prompt_hash": "same",
            "rejected_prompt_hash": "same",
            "rejected_candidates": 4,
            "protected_medical_terms": [],
        },
    }
    fidelity_skip = {
        "source_dialogue_id": "health",
        "turn_index": 1,
        "skip_reason": "sample_error",
        "error_message": "MediTOD翻訳で医療情報が失われました: citation",
        "source_prompt_hash": source_hash,
    }
    restored = restore_broad_resume_records(
        sources=[source],
        accepted=[],
        skipped=[],
        quarantined_accepted=[accepted],
        quarantined_skipped=[fidelity_skip],
        fidelity_retry=[],
        expected_generation_model="terra",
        expected_scoring_model="terra",
        expected_bayes_version="basis-v1",
        expected_candidates=4,
        min_score_gap=0.20,
        min_chosen_posterior=0.70,
        max_rejected_posterior=0.55,
    )
    assert len(restored["accepted"]) == 1
    assert restored["report"]["accepted_restored_from_personal_quarantine"] == 1
    assert restored["report"]["fidelity_errors_requeued"] == 0
    assert restored["accepted"][0]["medical_fidelity_version"].endswith(".v3")


def test_ranked_rescue_requires_safe_same_context_pair():
    row = {
        "skip_reason": "low_chosen",
        "prompt": "p",
        "chosen": "発症時期を教えてください。",
        "rejected": "分かりません。",
        "score_chosen": 0.68,
        "score_rejected": 0.40,
        "score_gap": 0.28,
        "metadata": {
            "translated_prompt_hash": "same",
            "rejected_prompt_hash": "same",
        },
    }
    assert rescue_eligible(row, min_chosen=0.60, max_rejected=0.65, min_gap=0.10)
    unsafe = {**row, "rejected": "Take 500 mg of this medicine."}
    assert not rescue_eligible(
        unsafe,
        min_chosen=0.60,
        max_rejected=0.65,
        min_gap=0.10,
    )


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


def test_meditod_gold_can_use_safe_shortfall_as_translation_pool():
    """gold 500件を満たす予備poolなら、上限未達でも保存できる。"""
    from tools.prepare_meditod_gold import collect_gold_candidates

    samples = []
    for index in range(3):
        samples.append(
            {
                "sample_id": f"s{index}",
                "conversation_id": f"c{index}",
                "history": [{"role": "user", "text": "咳があります。"}],
                "response": "いつからですか。",
                "next_user_turn": "昨日からです。",
                "metadata": {
                    "split": "train",
                    "dpo_eligible": True,
                    "ood": False,
                    "assistant_turn_index": 1,
                    "response_slots": ["symptom"],
                    "response_intents": ["question"],
                    "response_attributes": ["onset"],
                },
            }
        )

    rows = collect_gold_candidates(
        samples,
        target=5,
        seed=42,
        allow_target_shortfall=True,
        minimum_records=2,
    )
    assert len(rows) == 3
    with pytest.raises(ValueError, match="gold候補が不足"):
        collect_gold_candidates(
            samples,
            target=5,
            seed=42,
            allow_target_shortfall=True,
            minimum_records=4,
        )


def test_available_data_decision_requires_exhaustion_and_equal_arms(
    tmp_path: Path,
):
    accepted = tmp_path / "accepted.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    scored = tmp_path / "scored.jsonl"
    accepted.write_text(
        "\n".join(
            json.dumps(
                {
                    "source_dialogue_id": f"c{index}",
                    "turn_index": index,
                    "acceptance_rule": "strict",
                    "metadata": {
                        "translated_prompt_hash": f"h{index}",
                        "rejected_prompt_hash": f"h{index}",
                    },
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    candidates.write_text("{}\n{}\n{}\n", encoding="utf-8")
    scored.write_text("{}\n{}\n{}\n", encoding="utf-8")

    payload = validate_available_data_decision(
        accepted_path=accepted,
        candidates_path=candidates,
        scored_path=scored,
        basis_count=2,
        gold_count=1,
        random_count=3,
    )
    assert payload["source_exhausted"] is True
    assert payload["training_arms"]["basis_total"] == 3
    assert payload["training_arms"]["random_total"] == 3

    scored.write_text("{}\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="全件scoring"):
        validate_available_data_decision(
            accepted_path=accepted,
            candidates_path=candidates,
            scored_path=scored,
            basis_count=2,
            gold_count=1,
            random_count=3,
        )


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


def test_eval_translation_fidelity_accepts_japanese_medical_aliases():
    source = {
        "history_en": [
            {
                "role": "user",
                "text": (
                    "I have breathlessness, fatigue and swelling. "
                    "My temperature was 38 point, uh, point 5 C."
                ),
            },
            {
                "role": "assistant",
                "text": "Do you take metoprolol and lisinopril?",
            },
        ],
        "reference_response_en": "How long have you used insulin?",
        "next_patient_turn_en": "For 2 1/2 months.",
    }
    translated = {
        "history_ja": [
            {
                "role": "user",
                "text": "息切れ、疲労、むくみがあり、体温は38.5度でした。",
            },
            {
                "role": "assistant",
                "text": "メトプロロールとリシノプリルを服用していますか。",
            },
        ],
        "reference_response_ja": "インスリンはいつから使用していますか。",
        "next_patient_turn_ja": "2か月半です。",
    }
    assert evaluation_translation_fidelity_errors(source, translated) == {}


def test_eval_candidate_reserve_obeys_consultation_cap_and_fills_failure():
    config = load_yaml(FIXTURES / "meditod_public_raw_config.yaml")
    _, samples, _ = prepare_public_raw(
        FIXTURES / "meditod_dialogs.json",
        FIXTURES / "meditod_annotations.json",
        config=config,
        seed=42,
    )
    candidates, manifest = select_eval_prompt_candidates(
        samples,
        count=2,
        seed=42,
        ood=False,
        max_per_consultation=3,
        candidate_reserve=-1,
    )
    assert manifest["primary_count"] == 2
    assert len({row["prompt_id"] for row in candidates}) == len(candidates)
    assert max(Counter(row["conversation_id"] for row in candidates).values()) <= 3
    translated = []
    for row in candidates[1:]:
        translated.append(
            {
                **row,
                "history_ja": row["history_en"],
                "reference_response_ja": row["reference_response_en"],
                "next_patient_turn_ja": row["next_patient_turn_en"],
            }
        )
    final = finalize_translated_prompts(candidates, translated, count=2)
    assert len(final) == 2
    assert candidates[0]["prompt_id"] not in {
        row["prompt_id"] for row in final
    }
