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
    assert amendments == [
        {
            "amendment_id": "small_batch:fixture-fingerprint:3000",
            "timestamp": amendments[0]["timestamp"],
            "experiment_fingerprint": "fixture-fingerprint",
            "reason": "API評価の過剰実行を避けるため、保存済みscoringから判定batchだけを縮小",
            "original_scoring_batch_records": 20000,
            "continued_scoring_batch_records": 3000,
            "selection_pool_records": 5000,
            "starting_scored_records": 1,
            "models": {"scoring": "terra"},
            "scoring": {"preset": "mathdial_tutoring"},
            "selection": {"label_derivation_method": "state_specific_margin"},
        }
    ]
