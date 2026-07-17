"""MathDial v2再評価shellの軽量検査。"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mathdial_oracle_v2_reanalysis_shell_syntax():
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/run_mathdial_oracle_v2_reanalysis.sh")],
        check=True,
    )


def test_mathdial_oracle_v2_reanalysis_does_not_prepare_or_generate():
    script = (
        ROOT / "scripts/run_mathdial_oracle_v2_reanalysis.sh"
    ).read_text(encoding="utf-8")
    assert "tools.mathdial_evaluation prepare" not in script
    assert "tools.mathdial_evaluation generate" not in script
    assert "evaluation/oracle_input.jsonl" in script
    assert "evaluation/oracle/general/raw.jsonl" in script
    assert "post_hoc_reanalysis_on_v1_prompts_and_responses" in script
