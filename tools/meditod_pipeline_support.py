"""MediTOD pipelineのAPI/GPU不要mockと最終report支援。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from core.dpo_prompting import MEDITOD_DPO_PROMPT_TEMPLATE_VERSION, build_meditod_dpo_prompt_from_context_text
from tools.mathdial_pipeline_support import enrich_score_file, mock_score, read_jsonl, write_jsonl


def mock_dpo(rows: list[dict[str, Any]], *, count: int, source_dataset: str, gold: bool) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"MediTOD mock DPO入力不足: {len(rows)}/{count}")
    output = []
    for row in rows[:count]:
        context = f"User: [日本語訳] {row.get('prompt', '')}"
        prompt = build_meditod_dpo_prompt_from_context_text(context)
        digest = hashlib.sha256(context.encode()).hexdigest()
        output.append(
            {
                "prompt": prompt,
                "chosen": f"[日本語訳] {row.get('response', '症状について教えてください。')}",
                "rejected": "詳しいことは分かりません。",
                "score_chosen": 0.85,
                "score_rejected": 0.30,
                "score_gap": 0.55,
                "source_dataset": source_dataset,
                "source_dialogue_id": row.get("conversation_id"),
                "turn_index": row.get("turn_index", 0),
                "metadata": {
                    "source_dataset": source_dataset,
                    "gold": gold,
                    "translated_prompt_hash": digest,
                    "rejected_prompt_hash": digest,
                    "style_preset": "meditod_history_taking" if source_dataset != "WildChat-Random" else "general_quality",
                    "dpo_prompt_template": MEDITOD_DPO_PROMPT_TEMPLATE_VERSION,
                },
            }
        )
    return output


def build_report(root: Path) -> str:
    sections = [
        "# MediTOD × WildChat-1M run report",
        "",
        f"- run_root: `{root}`",
        "- target_style: systematic history taking",
        "- clinical_safety_claim: Oracle safety scores are proxy metrics, not clinical certification",
        "- public_raw_split: custom deterministic split; not the official canonical split",
        "",
    ]
    preferred = (
        "meditod/preprocessing_report.json",
        "basis_model/meditod_model_quality.json",
        "wildchat/statistics.json",
        "selections/selection_report.json",
        "evaluation/statistics/metadata.json",
        "evaluation/annotation_metrics_summary.json",
    )
    for relative in preferred:
        path = root / relative
        if not path.exists():
            continue
        sections.extend(
            [
                f"## {relative}",
                "",
                "```json",
                path.read_text(encoding="utf-8")[:12000].rstrip(),
                "```",
                "",
            ]
        )
    for name in ("model_summary.csv", "omnibus_friedman.csv", "posthoc_pairwise.csv", "cluster_omnibus_friedman.csv"):
        path = root / "evaluation" / "statistics" / name
        if path.exists():
            sections.extend([f"## evaluation/statistics/{name}", "", "```csv", path.read_text(encoding="utf-8")[:12000].rstrip(), "```", ""])
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD pipeline support")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("mock-score")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--bayes-model")
    enrich = sub.add_parser("enrich-score")
    enrich.add_argument("--input", required=True)
    enrich.add_argument("--output", required=True)
    enrich.add_argument("--skip-records", type=int, default=0)
    enrich.add_argument("--append", action="store_true")
    dpo = sub.add_parser("mock-dpo")
    dpo.add_argument("--input", required=True)
    dpo.add_argument("--output", required=True)
    dpo.add_argument("--count", type=int, required=True)
    dpo.add_argument("--source-dataset", required=True)
    dpo.add_argument("--gold", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "mock-score":
        write_jsonl(mock_score(read_jsonl(args.input), args.bayes_model), args.output)
    elif args.command == "enrich-score":
        enrich_score_file(Path(args.input), Path(args.output), skip_records=args.skip_records, append=args.append)
    elif args.command == "mock-dpo":
        write_jsonl(mock_dpo(read_jsonl(args.input), count=args.count, source_dataset=args.source_dataset, gold=args.gold), args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(build_report(args.root), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
