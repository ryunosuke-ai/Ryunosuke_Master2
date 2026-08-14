#!/usr/bin/env python3
"""Gemini/Claude Web用にモデルblindなOracle評価パケットを作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.oracle_eval_common import build_score_instructions, build_strategy_instructions
from scripts.eval_oracle_conversation_style_esconv_v2 import SPEC as ESCONV_STYLE
from scripts.eval_oracle_mathdial_v2 import CORE_V2 as MATHDIAL_PEDAGOGICAL
from scripts.eval_oracle_mathdial import GENERAL as MATHDIAL_GENERAL
from scripts.eval_oracle_meditod import GENERAL as MEDITOD_GENERAL
from scripts.eval_oracle_meditod import HISTORY as MEDITOD_HISTORY
from scripts.eval_oracle_meditod import SAFETY as MEDITOD_SAFETY
from scripts.eval_oracle_strategy_transition_esconv_v2 import SPEC as ESCONV_TRANSITION
from scripts.eval_oracle_tst import SPEC as ESCONV_TST


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/cross_model_oracle/web_packets_v1"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    path: Path
    categories: tuple[tuple[str, str, Any], ...]


SOURCES = (
    DatasetSource(
        name="esconv",
        path=ROOT / (
            "artifacts/evaluations/oracle_eval_runs/"
            "esconv_topconf_three_model_esconv_v2_100_gpt54_v1_"
            "topconf_three_model_esconv_v2_10pt/three_model_responses.jsonl"
        ),
        categories=(
            ("text_style_transfer", "score", ESCONV_TST),
            ("conversation_style", "score", ESCONV_STYLE),
            ("strategy_transition", "strategy", ESCONV_TRANSITION),
        ),
    ),
    DatasetSource(
        name="mathdial",
        path=ROOT / (
            "artifacts/mathdial_wildchat/evaluation_rechecks/"
            "mathdial_v6_instruction_outcome_selected_top100_v1/evaluation/oracle_input.jsonl"
        ),
        categories=(
            ("pedagogical_v2", "score", MATHDIAL_PEDAGOGICAL),
            ("general", "score", MATHDIAL_GENERAL),
        ),
    ),
    DatasetSource(
        name="meditod",
        path=ROOT / (
            "artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/"
            "evaluation/oracle_input.jsonl"
        ),
        categories=(
            ("history", "score", MEDITOD_HISTORY),
            ("general", "score", MEDITOD_GENERAL),
            ("safety", "score", MEDITOD_SAFETY),
        ),
    ),
)


MODEL_FIELDS = (
    ("base_response", "base"),
    ("bayes_dpo_response", "basis"),
    ("basis_response", "basis"),
    ("random_dpo_response", "random_dpo"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"評価入力がありません: {path}")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_history(value: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for turn in value if isinstance(value, list) else []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("speaker") or "unknown").strip()
        text = str(turn.get("text") or turn.get("content") or "").strip()
        if text:
            result.append({"role": role, "text": text})
    return result


def expand_source(source: DatasetSource) -> list[dict[str, Any]]:
    rows = read_jsonl(source.path)
    expanded: list[dict[str, Any]] = []
    if source.name == "esconv":
        for row in rows:
            item_id = str(row.get("prompt_id") or "").strip()
            if not item_id:
                raise ValueError("ESConv prompt_idが空です。")
            seen: set[str] = set()
            for field, model in MODEL_FIELDS:
                response = str(row.get(field) or "").strip()
                if not response or model in seen:
                    continue
                seen.add(model)
                expanded.append(
                    {
                        "item_id": item_id,
                        "model_name": model,
                        "prompt": str(row.get("prompt") or "").strip(),
                        "history": normalized_history(row.get("history")),
                        "response": response,
                    }
                )
    else:
        for row in rows:
            expanded.append(
                {
                    "item_id": str(row.get("sample_id") or "").strip(),
                    "model_name": str(row.get("model_name") or "").strip(),
                    "prompt": str(row.get("prompt") or "").strip(),
                    "history": normalized_history(row.get("history")),
                    "response": str(row.get("response") or "").strip(),
                }
            )
    if any(not row[key] for row in expanded for key in ("item_id", "model_name", "prompt", "response")):
        raise ValueError(f"{source.name}: 必須値が空の評価レコードがあります。")
    by_item: dict[str, set[str]] = {}
    for row in expanded:
        by_item.setdefault(row["item_id"], set()).add(row["model_name"])
    expected = {"base", "basis", "random_dpo"}
    invalid = {item: models for item, models in by_item.items() if models != expected}
    if invalid:
        raise ValueError(f"{source.name}: 3モデルが揃わないitemがあります: {list(invalid.items())[:3]}")
    if len(by_item) != 100 or len(expanded) != 300:
        raise ValueError(f"{source.name}: 100 item/300 responseではありません: {len(by_item)}/{len(expanded)}")
    return expanded


def blind_records(
    source: DatasetSource,
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, str]]]:
    by_item: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
    item_ids = sorted(by_item)
    rng = random.Random(f"{seed}:{source.name}")
    rng.shuffle(item_ids)
    waves: list[list[dict[str, Any]]] = [[], [], []]
    key_rows: list[dict[str, str]] = []
    for item_id in item_ids:
        candidates = list(by_item[item_id])
        rng.shuffle(candidates)
        for index, row in enumerate(candidates):
            blind_id = hashlib.sha256(
                f"{seed}:{source.name}:{item_id}:{index}".encode("utf-8")
            ).hexdigest()[:16]
            response_id = f"{source.name}_{blind_id}"
            public = {
                "item_id": item_id,
                "response_id": response_id,
                "prompt": row["prompt"],
                "history": row["history"],
                "response": row["response"],
            }
            waves[index].append(public)
            key_rows.append(
                {
                    "item_id": item_id,
                    "response_id": response_id,
                    "model_name": row["model_name"],
                }
            )
    return waves, key_rows


def batch_records(
    waves: list[list[dict[str, Any]]],
    *,
    max_records: int,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    for wave in waves:
        current: list[dict[str, Any]] = []
        current_chars = 0
        for row in wave:
            size = len(json.dumps(row, ensure_ascii=False))
            if current and (len(current) >= max_records or current_chars + size > max_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(row)
            current_chars += size
        if current:
            batches.append(current)
    return batches


def batch_protocol(axis_keys: list[str], *, strategy: bool) -> str:
    score_lines = ",\n".join(f'      "{key}": 1' for key in axis_keys)
    labels = (
        '    "labels": {"predicted_user_state_before_response": "...", '
        '"response_strategy": "...", "predicted_user_state_after_response": "...", '
        '"transition_type": "...", "ideal_strategy_for_context": "..."},\n'
        if strategy
        else ""
    )
    return f"""

## Webバッチ評価手順

別に添付されたJSONLには、1行につき1つの評価対象応答が入っています。
各行を互いに独立に評価し、他の行との相対比較や順位付けはしないでください。
同じitemの別モデル応答はこのバッチには含まれていません。モデルを推測しないでください。

入力の全行を元の順序で1回ずつ採点し、JSONLのみを返してください。
Markdown、コードブロック、前後の説明、集計、欠落行を含めないでください。
`overall_score`は`scores`の算術平均を小数で記録してください。

1行の出力schema:
{{
  "item_id": "入力と同じ値",
  "response_id": "入力と同じ値",
{labels}  "scores": {{
{score_lines}
  }},
  "overall_score": 1.0,
  "reason": "1〜2文の短い理由"
}}
""".strip()


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_dataset(
    source: DatasetSource,
    output_root: Path,
    *,
    seed: int,
    max_records: int,
    max_chars: int,
) -> dict[str, Any]:
    rows = expand_source(source)
    waves, answer_key = blind_records(source, rows, seed=seed)
    batches = batch_records(waves, max_records=max_records, max_chars=max_chars)
    dataset_dir = output_root / source.name
    for index, batch in enumerate(batches, start=1):
        write_jsonl(batch, dataset_dir / "inputs" / f"batch_{index:03d}.jsonl")
    write_jsonl(answer_key, dataset_dir / "private_answer_key.jsonl")

    prompt_files = []
    for category, kind, spec in source.categories:
        if kind == "strategy":
            base_prompt = build_strategy_instructions(score_scale=10, spec=spec)
            axes = [axis.key for axis in spec.score_axes]
        else:
            base_prompt = build_score_instructions(spec, score_scale=10)
            axes = [axis.key for axis in spec.axes]
        prompt = base_prompt + "\n\n" + batch_protocol(axes, strategy=kind == "strategy") + "\n"
        path = dataset_dir / "prompts" / f"{category}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt, encoding="utf-8")
        prompt_files.append(
            {"category": category, "path": str(path), "sha256": sha256_file(path)}
        )

    manifest = {
        "dataset": source.name,
        "source": str(source.path),
        "source_sha256": sha256_file(source.path),
        "seed": seed,
        "items": 100,
        "responses": 300,
        "batches": len(batches),
        "max_records_per_batch": max_records,
        "max_chars_per_batch": max_chars,
        "model_identity_removed_from_public_inputs": True,
        "same_item_responses_separated_between_waves": True,
        "prompt_files": prompt_files,
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max-records", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=100_000)
    args = parser.parse_args()
    if args.max_records < 1 or args.max_chars < 10_000:
        raise ValueError("batch上限が小さすぎます。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [
        prepare_dataset(
            source,
            args.output_dir,
            seed=args.seed,
            max_records=args.max_records,
            max_chars=args.max_chars,
        )
        for source in SOURCES
    ]
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"datasets": manifests}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Web Oracle評価パケットを書き出しました: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
