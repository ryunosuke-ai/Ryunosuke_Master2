from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_small_batch_resume_records_auditable_amendment(tmp_path: Path):
    run_root = tmp_path / "run"
    (run_root / "scoring").mkdir(parents=True)
    (run_root / "basis_model").mkdir(parents=True)
    (run_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_tag": "fixture-run",
                "experiment_fingerprint": "fixture-fingerprint",
                "models": {"scoring": "terra"},
                "scoring": {"preset": "mathdial_tutoring"},
                "selection": {"label_derivation_method": "state_specific_margin"},
                "early_stop": {"scoring_batch_records": 20000},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "scoring/wildchat_scored_raw.jsonl").write_text(
        '{"conversation_id":"c1"}\n', encoding="utf-8"
    )
    (run_root / "scoring/prioritized_candidates.jsonl").write_text(
        '{"conversation_id":"c1"}\n', encoding="utf-8"
    )
    (run_root / "basis_model/mathdial_transition_compat.json").write_text(
        "{}\n", encoding="utf-8"
    )
    environment = {
        **os.environ,
        "RUN_TAG": "fixture-run",
        "OUTPUT_ROOT": str(run_root),
        "SCORING_BATCH_RECORDS": "3000",
        "ORIGINAL_SCORING_BATCH_RECORDS": "20000",
        "SELECTION_POOL_COUNT": "5000",
        "MATHDIAL_SCORING_LLM_MODEL": "terra",
        "SCORING_PRESET": "mathdial_tutoring",
        "SMALL_BATCH_VALIDATE_ONLY": "1",
    }
    completed = subprocess.run(
        [str(PROJECT_ROOT / "scripts/resume_mathdial_scoring_small_batches.sh")],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    amendment_path = run_root / "scoring/scoring_configuration_amendments.jsonl"
    amendments = [json.loads(line) for line in amendment_path.open(encoding="utf-8")]
    assert len(amendments) == 1
    amendment = amendments[0]
    assert amendment["amendment_id"] == (
        "length_bounded_v1:fixture-fingerprint:3000:16000:6144"
    )
    assert amendment["experiment_fingerprint"] == "fixture-fingerprint"
    assert amendment["original_scoring_batch_records"] == 20000
    assert amendment["continued_scoring_batch_records"] == 3000
    assert amendment["selection_pool_records"] == 5000
    assert amendment["length_eligibility"] == {
        "max_source_characters": 16000,
        "policy": "exclude_whole_sample_without_truncating_history",
    }
    assert amendment["dpo_max_output_tokens"] == 6144
    assert amendment["starting_scored_records"] == 1
    assert amendment["mandatory_fallback_repair"] is False
    assert amendment["exclude_fallback_conversations_from_basis"] is True
    assert amendment["models"] == {"scoring": "terra"}
    assert amendment["scoring"] == {"preset": "mathdial_tutoring"}
    assert amendment["selection"] == {
        "label_derivation_method": "state_specific_margin"
    }
    assert amendment["runtime_rate_limit"] == {
        "scoring_requests_per_minute": 120.0,
        "repair_requests_per_minute": 90.0,
        "max_retries": 6,
        "initial_backoff_seconds": 15.0,
    }
