from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.mathdial_basis import (
    build_basis_model,
    build_extraction_instructions,
    build_transition_compat_model,
    load_yaml,
    score_extraction,
    validate_extraction,
    validate_ontology,
)
from core.transition_bayes_model import parse_transition_bayes_model
from scripts.eval_oracle_mathdial import GENERAL, PEDAGOGICAL
from scripts.run_mathdial_statistics import analyze
from tools.mathdial_evaluation import blind_oracle_rows, select_test_prompts, validate_translation
from tools.mathdial_features import normalize_extraction_payload, quality_metrics, validate_with_llm
from tools.mathdial_pipeline_support import mock_dpo, mock_score
from tools.mathdial_selection import select_groups
from tools.prepare_mathdial_for_analysis import (
    select_analysis_conversations,
    summarize as summarize_analysis_corpus,
    to_analysis_record,
)
from tools.analyze_mathdial_corpus_transition_bayes import (
    build_mathdial_corpus_text,
    generate_model as generate_mathdial_model,
    mock_model as mock_mathdial_model,
)
from tools.mix_mathdial_dpo import build_training_arms, validate_pair
from tools.translate_and_generate_dpo import _style_specific_translation_policy
from tools.train_qwen35_dpo_lora import dpo_loss_from_logps
from tools.score_dialogue_with_transition_bayes_model import limit_records_by_conversation
from tools.wildchat_tutoring import (
    domain_flags,
    extract_candidates,
    normalize_wildchat_row,
    sample_to_scoring_record,
)
from core.dialogue_schema import build_assistant_samples


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "configs/ontologies/mathdial_v1.yaml"
WILD_CONFIG_PATH = ROOT / "configs/datasets/wildchat_tutoring.yaml"


def extraction_row(**updates):
    row = {
        "sample_id": "c1#assistant-0001",
        "conversation_id": "c1",
        "split": "train",
        "assistant_turn_index": 1,
        "student_state_before": "misconception",
        "tutor_strategy": "probing_question",
        "student_state_after": "partial_understanding",
        "conversation_stage": "guided_reasoning",
        "style_features": ["elicits_reasoning", "withholds_final_answer"],
        "confidence": 0.9,
        "short_reason": "The tutor asks the learner to inspect the mistaken step.",
        "validation_status": "valid",
        "teacher_moves": ["probing"],
    }
    row.update(updates)
    return row


def test_extraction_payload_normalizes_numeric_confidence_string():
    payload = normalize_extraction_payload({"confidence": "0.85"})
    assert payload["confidence"] == pytest.approx(0.85)


def test_ontology_schema_and_teacher_move_mapping_are_valid():
    ontology = validate_ontology(load_yaml(ONTOLOGY_PATH))
    assert ontology["version"] == "mathdial_v1"
    assert set(ontology["teacher_move_mapping"]) == {"probing", "focus", "telling", "generic"}


def test_extraction_prompt_does_not_expose_teacher_move_labels():
    prompt = build_extraction_instructions(load_yaml(ONTOLOGY_PATH)).lower()
    assert "annotated_strategy" not in prompt
    assert "teacher move label" not in prompt


def test_extraction_schema_rejects_unknown_label():
    ontology = load_yaml(ONTOLOGY_PATH)
    payload = extraction_row()
    payload["tutor_strategy"] = "unknown"
    with pytest.raises(ValueError, match="未知ラベル"):
        validate_extraction(payload, ontology)


def test_secondary_validation_mock_marks_record_valid():
    ontology = load_yaml(ONTOLOGY_PATH)
    sample = {"sample_id": "c1#assistant-0001", "conversation_id": "c1", "history": [{"role": "user", "text": "2+2=5"}], "response": "Can you check that sum?", "next_user_turn": "It is 4.", "metadata": {}}
    conversation = {"conversation_id": "c1", "metadata": {"question": "2+2?", "ground_truth": "4"}}
    output, errors = validate_with_llm([extraction_row(validation_status="unvalidated")], [sample], [conversation], ontology, generator=None, model="mock", mock=True, mode="all")
    assert not errors
    assert output[0]["validation_status"] == "valid"


def test_quality_metrics_accept_multi_value_teacher_move():
    metrics = quality_metrics([extraction_row(teacher_moves=["focus", "probing"])], load_yaml(ONTOLOGY_PATH))
    assert metrics["ambiguity_aware_accuracy"] == 1.0
    assert metrics["single_label_evaluated"] == 0


def test_basis_model_and_esconv_compatible_view_round_trip():
    ontology = load_yaml(ONTOLOGY_PATH)
    rows = [
        extraction_row(),
        extraction_row(sample_id="c1#assistant-0003", assistant_turn_index=3, tutor_strategy="scaffolded_hint", conversation_stage="focused_scaffolding"),
        extraction_row(sample_id="c1#assistant-0005", assistant_turn_index=5, tutor_strategy="comprehension_check", student_state_before="partial_understanding", student_state_after="corrected_understanding", conversation_stage="verification"),
    ]
    model = build_basis_model(rows, ontology)
    scored = score_extraction(rows[0], model)
    assert 0 <= scored["basis_score"] <= 1
    compat = build_transition_compat_model(rows, ontology)
    parsed = parse_transition_bayes_model(compat)
    assert parsed.model_type == "transition_bayes_network"
    assert "scaffolding" in parsed.positive_states


def _analysis_conversation(index: int, move: str, *, split: str = "train"):
    return {
        "conversation_id": f"c{index:03d}",
        "source_dataset": "mathdial",
        "split": split,
        "turns": [
            {
                "role": "assistant",
                "text": f"Question {index}",
                "metadata": {"teacher_moves": [move]},
            },
            {"role": "user", "text": f"Attempt {index}"},
        ],
        "metadata": {
            "qid": f"q{index:03d}",
            "question": f"Problem {index}",
            "ground_truth": str(index),
        },
    }


def test_direct_basis_analysis_sample_is_train_only_unique_and_reproducible():
    moves = ("probing", "focus", "telling", "generic")
    records = [_analysis_conversation(index, moves[index % 4]) for index in range(100)]
    records.append(_analysis_conversation(100, "probing", split="test"))
    first = select_analysis_conversations(records, count=80, seed=42)
    second = select_analysis_conversations(list(reversed(records)), count=80, seed=42)
    assert [row["conversation_id"] for row in first] == [row["conversation_id"] for row in second]
    assert len(first) == len({row["metadata"]["qid"] for row in first}) == 80
    assert {row["split"] for row in first} == {"train"}
    analysis = [to_analysis_record(row) for row in first]
    assert all(summarize_analysis_corpus(analysis)["teacher_moves"].values())


def test_teacher_move_is_analysis_annotation_not_dialogue_body():
    source = _analysis_conversation(1, "probing")
    analysis = to_analysis_record(source)
    text = build_mathdial_corpus_text([analysis])
    assert source["turns"][0]["text"] == "Question 1"
    assert "(probing)" not in source["turns"][0]["text"]
    assert 'annotated_teacher_moves=["probing"]' in text


class _RepairGenerator:
    def __init__(self):
        self.outputs = ["not-json", json.dumps(mock_mathdial_model())]

    def generate(self, **_kwargs):
        return self.outputs.pop(0)


def test_direct_basis_generation_repairs_json_once():
    record = to_analysis_record(_analysis_conversation(1, "probing"))
    payload, _, _ = generate_mathdial_model(
        [record],
        generator=_RepairGenerator(),
        model="mock-sol",
        max_output_tokens=1000,
        max_input_chars=10000,
        mock=False,
    )
    assert parse_transition_bayes_model(payload).model_type == "transition_bayes_network"


def test_direct_basis_rejects_schema_invalid_model():
    invalid = mock_mathdial_model()
    invalid["initial_state_prior"][invalid["states"][0]] = 0.99
    generator = _RepairGenerator()
    generator.outputs = [json.dumps(invalid)]
    with pytest.raises(ValueError):
        generate_mathdial_model(
            [to_analysis_record(_analysis_conversation(1, "probing"))],
            generator=generator,
            model="mock-sol",
            max_output_tokens=1000,
            max_input_chars=10000,
            mock=False,
        )


def valid_wildchat_row():
    return {
        "language": "English",
        "toxic": False,
        "redacted": False,
        "model": "gpt-4",
        "conversation": [
            {"role": "user", "content": "I am learning math and I am confused."},
            {"role": "assistant", "content": "Show your first step."},
            {"role": "user", "content": "I added incorrectly."},
            {"role": "assistant", "content": "Which terms did you add?"},
            {"role": "user", "content": "Two and three."},
            {"role": "assistant", "content": "What is their sum?"},
        ],
    }


def test_wildchat_normalization_domain_turn_filter_and_pii_minimization():
    config = load_yaml(WILD_CONFIG_PATH)
    record, reason = normalize_wildchat_row(valid_wildchat_row(), config)
    assert reason is None
    assert record["num_assistant_turns"] == 3
    assert domain_flags(record, config)["general"]
    assert set(record["metadata"]) == {"conversation_hash", "source_model", "eligible_for_training"}
    samples = build_assistant_samples(record)
    scoring = sample_to_scoring_record(samples[-1])
    assert scoring["history"] == samples[-1]["history"]
    assert scoring["next_user_turn"] is None


def test_wildchat_rejects_toxic_short_and_near_duplicate():
    config = load_yaml(WILD_CONFIG_PATH)
    toxic = valid_wildchat_row()
    toxic["toxic"] = True
    assert normalize_wildchat_row(toxic, config)[1] == "toxic"
    general, _, stats = extract_candidates([valid_wildchat_row(), valid_wildchat_row()], config)
    assert len(general) == 1
    assert stats["excluded_exact_duplicate"] == 1


def test_wildchat_stops_after_candidate_target_at_conversation_boundary():
    config = {**load_yaml(WILD_CONFIG_PATH), "near_duplicate_jaccard": 1.0}
    rows = []
    for index in range(4):
        row = valid_wildchat_row()
        row["conversation"][0]["content"] += f" Example {index}."
        rows.append(row)
    general, _, stats = extract_candidates(
        rows,
        config,
        target_candidate_records=4,
        progress_every=0,
    )
    assert len(general) == 2
    assert stats["general_candidate_records"] == 6
    assert stats["stopped_by_candidate_target"] == 1
    assert stats["stream_rows"] == 2


def test_wildchat_checkpoint_resume_does_not_duplicate_candidates():
    config = {**load_yaml(WILD_CONFIG_PATH), "near_duplicate_jaccard": 1.0}
    rows = []
    for index in range(4):
        row = valid_wildchat_row()
        row["conversation"][0]["content"] += f" Resume example {index}."
        rows.append(row)
    checkpoints = []
    first_general, first_math, first_stats = extract_candidates(
        rows[:2],
        config,
        progress_every=0,
        checkpoint_every=1,
        on_checkpoint=lambda general, math, counts, completed: checkpoints.append(
            (len(general), len(math), counts["stream_rows"], completed)
        ),
    )
    general, _, stats = extract_candidates(
        rows[2:],
        config,
        progress_every=0,
        initial_general=first_general,
        initial_math=first_math,
        initial_counts=first_stats,
    )
    assert checkpoints
    assert len(general) == 4
    assert len({row["conversation_id"] for row in general}) == 4
    assert stats["stream_rows"] == 4


def test_scoring_limit_preserves_complete_conversations():
    records = [
        {"conversation_id": conversation_id, "turn_index": turn_index}
        for conversation_id, size in (("c1", 2), ("c2", 3), ("c3", 2))
        for turn_index in range(size)
    ]
    limited = limit_records_by_conversation(records, 4)
    assert len(limited) == 4
    assert {row["conversation_id"] for row in limited} == {"c1", "c3"}
    assert len(limit_records_by_conversation(records, None)) == len(records)


def test_scoring_pilot_includes_first_conversation_crossing_minimum():
    records = [
        {"conversation_id": conversation_id, "turn_index": turn_index}
        for conversation_id, size in (("c1", 2), ("c2", 3), ("c3", 2))
        for turn_index in range(size)
    ]
    limited = limit_records_by_conversation(
        records,
        4,
        include_crossing_conversation=True,
    )
    assert len(limited) == 5
    assert {row["conversation_id"] for row in limited} == {"c1", "c2"}


def test_selection_builds_three_groups_with_esconv_selector():
    scored = mock_score([
        {"sample_id": f"s{i}", "conversation_id": f"c{i}", "turn_index": 1, "prompt": "learn math", "response": f"response {i}", "metadata": {"context_turns": 3}}
        for i in range(8)
    ])
    mathdial = [{"split": "train", "metadata": {"question": "math equation"}}]
    groups = select_groups(scored, mathdial, count=3, random_count=4, seed=42)
    assert {key: len(value) for key, value in groups.items()} == {"domain_random": 4, "topic_similarity_top": 3, "basis_top": 3}


def test_selection_uses_generated_state_and_observation_names(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(mock_mathdial_model()), encoding="utf-8")
    source = [
        {"sample_id": f"s{i}", "conversation_id": f"c{i}", "turn_index": 1, "prompt": "learn math", "response": f"response {i}", "metadata": {"context_turns": 3}}
        for i in range(8)
    ]
    scored = mock_score(source, model_path)
    groups = select_groups(
        scored,
        [{"split": "train", "metadata": {"question": "math equation"}}],
        count=3,
        random_count=4,
        seed=42,
        bayes_model_path=model_path,
    )
    assert len(groups["basis_top"]) == 3
    assert {row["most_likely_state"] for row in groups["basis_top"]} <= set(mock_mathdial_model()["states"])


def test_mathdial_translation_policy_preserves_errors_and_does_not_enhance():
    policy = _style_specific_translation_policy("mathdial_tutoring")
    assert "誤答を翻訳時に訂正しない" in policy
    assert "BASiSらしさを翻訳によって強めない" in policy


def test_dpo_same_context_and_esconv_training_composition():
    basis = mock_dpo([{"conversation_id": f"b{i}", "turn_index": 1, "prompt": "p", "response": "c"} for i in range(2)], count=2, source_dataset="WildChat-BASiS", gold=False)
    gold = mock_dpo([{"conversation_id": "g", "turn_index": 1, "prompt": "p", "response": "c"}], count=1, source_dataset="MathDial", gold=True)
    random_rows = mock_dpo([{"conversation_id": f"r{i}", "turn_index": 1, "prompt": "p", "response": "c"} for i in range(3)], count=3, source_dataset="WildChat-Random", gold=False)
    validate_pair(basis[0])
    basis_arm, random_arm = build_training_arms(basis, gold, random_rows, basis_count=2, gold_count=1, random_count=3)
    assert len(basis_arm) == len(random_arm) == 3
    assert sum(row["metadata"]["gold"] for row in basis_arm) == 1
    assert not any(row["metadata"]["gold"] for row in random_arm)
    assert dpo_loss_from_logps([-1.0], [-3.0], beta=0.1) > 0


def test_evaluation_translation_preserves_roles_and_test_selection_is_unique():
    conversations = [{"conversation_id": "c1", "split": "test", "metadata": {"qid": "q1", "question": "2+2?", "ground_truth": "4"}}]
    samples = [{"sample_id": "s1", "conversation_id": "c1", "history": [{"role": "user", "text": "5"}], "metadata": {"split": "test", "history_ends_with_user": True, "teacher_moves": ["probing"]}}]
    selected = select_test_prompts(samples, conversations, count=1, seed=42)
    translated = validate_translation(selected[0], {"problem_ja": "2+2は?", "ground_truth_ja": "4", "history_ja": [{"role": "user", "text": "5"}]})
    assert translated["history_ja"][0]["role"] == "user"


def test_oracle_specs_are_frozen_and_model_blind():
    assert [axis.key for axis in PEDAGOGICAL.axes[:5]] == ["tutoring_style_strength", "misconception_diagnosis", "scaffolding_quality", "premature_answer_avoidance", "pedagogical_transition_plausibility"]
    text = " ".join(axis.description + axis.high + axis.low for axis in (*PEDAGOGICAL.axes, *GENERAL.axes)).lower()
    assert "basis-dpo" not in text and "random-dpo" not in text
    grouped = [{"sample_id": "s", "response_order": ["random_dpo", "base", "basis"], "problem_ja": "問題", "ground_truth_ja": "答え", "problem_en": "problem", "ground_truth_en": "answer", "history": [], "base_response": "b", "basis_response": "x", "random_dpo_response": "r"}]
    blind = blind_oracle_rows(grouped)
    assert [row["model_name"] for row in blind] == ["random_dpo", "base", "basis"]
    assert all("basis" not in row["prompt"].lower() for row in blind)


def test_statistics_runs_friedman_and_conditional_posthoc():
    data = {"pedagogical.axis": {f"s{i}": {"base": 3.0, "basis": 9.0, "random_dpo": 4.0} for i in range(8)}}
    summary, omnibus, posthoc = analyze(data, permutations=100, bootstrap=100, seed=42)
    assert len(summary) == 3
    assert omnibus[0]["basis_highest"] == "true"
    assert len(posthoc) == 3


def test_shell_scripts_have_valid_syntax():
    for script in ("scripts/run_mathdial_wildchat_pipeline.sh", "scripts/run_mathdial_wildchat_dry_run.sh", "scripts/run_mathdial_wildchat_watchdog.sh"):
        subprocess.run(["bash", "-n", str(ROOT / script)], check=True)


def test_watchdog_terminates_stall_and_restarts(tmp_path):
    fake = tmp_path / "fake_pipeline.sh"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$OUTPUT_ROOT\"\n"
        "printf '{\"stage\":\"score_wildchat\"}\\n' > \"$OUTPUT_ROOT/pipeline_status.json\"\n"
        "if [[ \"${WATCHDOG_ATTEMPT:-1}\" == \"1\" ]]; then sleep 30; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = {
        "PATH": "/usr/bin:/bin",
        "RUN_TAG": "watchdog_test",
        "OUTPUT_ROOT": str(tmp_path / "run"),
        "MATHDIAL_PIPELINE_SCRIPT": str(fake),
        "WATCHDOG_INTERVAL_SECONDS": "1",
        "WATCHDOG_STALL_SECONDS": "1",
        "WATCHDOG_KILL_GRACE_SECONDS": "1",
        "WATCHDOG_MAX_RESTARTS": "2",
    }
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/run_mathdial_wildchat_watchdog.sh")],
        env=env,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0
    log = (tmp_path / "run/watchdog/watchdog.log").read_text(encoding="utf-8")
    assert "進捗停止を検出" in log
    assert "attempt=2" in log


def test_same_run_tag_rejects_changed_experiment_fingerprint(tmp_path):
    root = tmp_path / "fingerprint"
    base_env = {
        "PATH": "/usr/bin:/bin",
        "RUN_TAG": "fingerprint_test",
        "OUTPUT_ROOT": str(root),
        "DRY_RUN": "1",
        "STAGE": "report",
        "SEED": "42",
    }
    script = str(ROOT / "scripts/run_mathdial_wildchat_pipeline.sh")
    first = subprocess.run(["bash", script], cwd=ROOT, env=base_env, check=False)
    changed = subprocess.run(
        ["bash", script],
        cwd=ROOT,
        env={**base_env, "SEED": "43"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    assert changed.returncode != 0
    assert "同じRUN_TAGの実験条件が変わっています" in changed.stderr + changed.stdout
