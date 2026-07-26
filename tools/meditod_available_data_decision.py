"""全候補処理後のMediTOD学習件数決定を監査して保存する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DECISION_VERSION = "meditod_available_selected_gold500.v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_available_data_decision(
    *,
    accepted_path: Path,
    candidates_path: Path,
    scored_path: Path,
    basis_count: int,
    gold_count: int,
    random_count: int,
) -> dict[str, Any]:
    if min(basis_count, gold_count, random_count) <= 0:
        raise ValueError("MediTOD学習件数は正の整数にしてください。")
    if random_count != basis_count + gold_count:
        raise ValueError(
            "MediTOD Random件数はBASiS選別件数とgold件数の合計にしてください。"
        )

    candidate_count = count_jsonl(candidates_path)
    scored_count = count_jsonl(scored_path)
    if candidate_count != scored_count:
        raise ValueError(
            "MediTOD広域候補を全件scoringしていません: "
            f"{scored_count}/{candidate_count}"
        )

    accepted = read_jsonl(accepted_path)
    if len(accepted) != basis_count:
        raise ValueError(
            "MediTOD採択済み件数が移行条件と一致しません: "
            f"{len(accepted)}/{basis_count}"
        )

    keys: list[tuple[str, int]] = []
    for row in accepted:
        keys.append(
            (str(row.get("source_dialogue_id", "")), int(row.get("turn_index", -1)))
        )
        metadata = row.get("metadata", {})
        if (
            not metadata.get("translated_prompt_hash")
            or metadata.get("translated_prompt_hash")
            != metadata.get("rejected_prompt_hash")
        ):
            raise ValueError("MediTOD採択済みDPOにcontext hash不一致があります。")
        acceptance_rule = row.get("acceptance_rule") or metadata.get(
            "acceptance_rule"
        )
        if acceptance_rule != "strict":
            raise ValueError(
                "今回の固定学習集合にはstrict採択以外を含められません。"
            )
    if len(set(keys)) != len(keys):
        raise ValueError("MediTOD採択済みDPOに重複があります。")

    return {
        "decision_version": DECISION_VERSION,
        "reason": "all_broad_health_candidates_exhausted_before_original_target",
        "source_exhausted": True,
        "training_arms": {
            "basis_selected": basis_count,
            "meditod_gold": gold_count,
            "basis_total": basis_count + gold_count,
            "random_total": random_count,
            "random_gold": 0,
        },
        "source_counts": {
            "broad_health_candidates": candidate_count,
            "scored_candidates": scored_count,
            "strict_accepted": len(accepted),
        },
        "checks": {
            "all_candidates_scored": True,
            "accepted_unique": True,
            "same_context_hash": True,
            "strict_acceptance_only": True,
            "equal_training_arm_size": True,
        },
        "files": {
            "accepted": {
                "path": str(accepted_path),
                "sha256": file_sha256(accepted_path),
            },
            "candidates": {
                "path": str(candidates_path),
                "sha256": file_sha256(candidates_path),
            },
            "scored": {
                "path": str(scored_path),
                "sha256": file_sha256(scored_path),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediTODの利用可能なstrict採択データで学習件数を固定"
    )
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basis-count", type=int, required=True)
    parser.add_argument("--gold-count", type=int, required=True)
    parser.add_argument("--random-count", type=int, required=True)
    args = parser.parse_args()

    payload = validate_available_data_decision(
        accepted_path=Path(args.accepted),
        candidates_path=Path(args.candidates),
        scored_path=Path(args.scored),
        basis_count=args.basis_count,
        gold_count=args.gold_count,
        random_count=args.random_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["training_arms"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
