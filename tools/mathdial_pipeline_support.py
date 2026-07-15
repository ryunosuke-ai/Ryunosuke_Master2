"""MathDial pipelineのAPI/GPU不要dry-runと結果reportを支援する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.transition_bayes_model import load_transition_bayes_model
from tools.extract_high_posterior_dialogues import derive_selection_labels_from_model


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def mock_score(
    rows: list[dict[str, Any]], bayes_model_path: Path | str | None = None
) -> list[dict[str, Any]]:
    output = []
    if bayes_model_path:
        model = load_transition_bayes_model(bayes_model_path)
        labels = derive_selection_labels_from_model(model)
        observations = tuple(filter(None, labels["prefer_observations"].split(",")))
        states = model.positive_states
        all_states = model.states
    else:
        observations = ("probing_question", "focusing_question", "scaffolded_hint", "explanation")
        states = ("diagnosing", "scaffolding", "explaining", "verifying")
        all_states = (*states, "premature_telling", "generic_ungrounded")
    if not observations or not states:
        raise ValueError("mock scoreに利用できるpositive state/observationがありません。")
    for index, row in enumerate(rows):
        observation = observations[index % len(observations)]
        state = states[index % len(states)]
        posterior = 0.65 + 0.05 * (index % 6)
        weights = {name: 0.7 if name == state else 0.1 / max(1, len(all_states) - 1) for name in all_states}
        output.append({**row, "observation": observation, "observation_score": 0.8, "reason": "mock scoring", "prior": 0.5, "posterior": min(0.95, posterior), "delta": posterior - 0.5, "state_posteriors": weights, "most_likely_state": state})
    return output


def enrich_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """既存ESConv scorer出力をMathDial診断schemaへ補完する。"""
    output = []
    for row in rows:
        delta = float(row.get("delta", 0.0))
        output.append({**row, "basis_score": float(row.get("posterior", 0.0)), "state_score": float(row.get("prior", 0.0)), "strategy_score": float(row.get("observation_score", 0.0)), "transition_score": max(0.0, min(1.0, 0.5 + delta)), "style_score": float(row.get("observation_score", 0.0)), "predicted_state_before": max(row.get("prior_state_distribution", {}) or {"unobserved": 1.0}, key=(row.get("prior_state_distribution", {}) or {"unobserved": 1.0}).get), "predicted_strategy": row.get("observation"), "predicted_state_after": row.get("most_likely_state"), "predicted_stage": row.get("most_likely_state"), "short_reason": row.get("reason", "")})
    return output


def enrich_score_file(
    input_path: Path,
    output_path: Path,
    *,
    skip_records: int = 0,
    append: bool = False,
) -> int:
    """巨大なscoring JSONLを全件メモリへ載せず診断schemaへ補完する。"""
    if skip_records < 0:
        raise ValueError("skip_recordsは0以上にしてください。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    if append:
        destination = output_path
        mode = "a"
    else:
        destination = output_path.with_suffix(output_path.suffix + ".tmp")
        mode = "w"
    try:
        with input_path.open(encoding="utf-8") as source, destination.open(
            mode, encoding="utf-8"
        ) as target:
            record_index = 0
            for line in source:
                if not line.strip():
                    continue
                if record_index < skip_records:
                    record_index += 1
                    continue
                row = json.loads(line)
                enriched = enrich_scores([row])[0]
                target.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                written += 1
                record_index += 1
        if not append:
            destination.replace(output_path)
    except Exception:
        if not append:
            destination.unlink(missing_ok=True)
        raise
    return written


def mock_dpo(rows: list[dict[str, Any]], *, count: int, source_dataset: str, gold: bool) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"mock DPO入力不足: {len(rows)}/{count}")
    output = []
    for row in rows[:count]:
        prompt_ja = f"User: [日本語訳] {row.get('prompt', '')}"
        chosen = f"[日本語訳] {row.get('response', '考え方を確認しましょう。')}"
        rejected = "答えだけを確認してください。"
        digest = hashlib.sha256(prompt_ja.encode()).hexdigest()
        output.append({"prompt": f"以下の会話の次のAI返答を生成してください。\n\n{prompt_ja}\n\nAI:", "chosen": chosen, "rejected": rejected, "score_chosen": 0.8, "score_rejected": 0.3, "score_gap": 0.5, "source_dataset": source_dataset, "source_dialogue_id": row.get("conversation_id"), "turn_index": row.get("turn_index", 0), "metadata": {"source_dataset": source_dataset, "gold": gold, "translated_prompt_hash": digest, "rejected_prompt_hash": digest, "style_preset": "mathdial_tutoring" if source_dataset != "WildChat-Random" else "general_quality"}})
    return output


def build_report(root: Path) -> str:
    sections = ["# MathDial × WildChat-1M run report", "", f"- run_root: `{root}`", ""]
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_SUCCESS"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sections.extend([f"## {path.relative_to(root)}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2)[:8000], "```", ""])
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial pipeline support")
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
        written = enrich_score_file(
            Path(args.input),
            Path(args.output),
            skip_records=args.skip_records,
            append=args.append,
        )
        print(f"[enrich-score] written={written} append={args.append}", flush=True)
    elif args.command == "mock-dpo":
        write_jsonl(mock_dpo(read_jsonl(args.input), count=args.count, source_dataset=args.source_dataset, gold=args.gold), args.output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(build_report(args.root), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
