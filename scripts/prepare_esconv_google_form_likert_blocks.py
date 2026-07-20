#!/usr/bin/env python3
"""ESConv Likert評価20件を、カテゴリ均衡した2実験へ分割する。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_esconv_google_form_eval import (  # noqa: E402
    DEFAULT_TOPCONF_RUN,
    DEFAULT_V2_RUN,
    MODEL_KEYS,
    build_candidates,
    load_axis_scores,
    select_oracle_enriched,
    sha256_file,
    version_orders,
    write_json,
    write_jsonl,
)
from scripts.prepare_esconv_google_form_likert_eval import (  # noqa: E402
    DEFAULT_SEED,
    LIKERT_ANCHORS,
    LIKERT_STATEMENTS,
    private_record,
    public_record,
    selection_diagnostics,
    write_apps_script,
    write_csv,
    write_markdown,
)


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/user_eval/google_forms/esconv_oracle_enriched_likert_blocked_v3"
)
EXPERIMENTS = ("A", "B")
COUNTERBALANCE_VERSIONS = ("1", "2", "3")


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="ESConv Likert評価20件を10件ずつの実験A/Bへ分割します。"
    )
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_V2_RUN)
    parser.add_argument("--topconf-run", type=Path, default=DEFAULT_TOPCONF_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--diagnostic-permutations", type=int, default=100_000)
    return parser.parse_args()


def split_category_pairs(
    selected: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """各カテゴリを1件ずつ含み、Oracle優位度も近い2実験へ分ける。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(row["category"], []).append(row)
    invalid = {
        category: len(rows)
        for category, rows in grouped.items()
        if len(rows) != 2
    }
    if invalid:
        raise ValueError(
            "2ブロック分割には各カテゴリ2件が必要です: "
            f"{dict(sorted(invalid.items()))}"
        )

    categories = sorted(grouped)
    pairs = [
        sorted(
            grouped[category],
            key=lambda row: (
                row["basis_advantage_over_best_control"],
                row["prompt_id"],
            ),
            reverse=True,
        )
        for category in categories
    ]
    best_assignment: tuple[float, tuple[int, ...]] | None = None
    for choices in itertools.product((0, 1), repeat=len(categories)):
        advantage_a = sum(
            pair[choice]["basis_advantage_over_best_control"]
            for pair, choice in zip(pairs, choices)
        )
        advantage_b = sum(
            pair[1 - choice]["basis_advantage_over_best_control"]
            for pair, choice in zip(pairs, choices)
        )
        objective = abs(advantage_a - advantage_b)
        candidate = (objective, choices)
        if best_assignment is None or candidate < best_assignment:
            best_assignment = candidate
    if best_assignment is None:
        raise ValueError("分割対象が空です。")

    choices = best_assignment[1]
    experiment_a = [pair[choice] for pair, choice in zip(pairs, choices)]
    experiment_b = [pair[1 - choice] for pair, choice in zip(pairs, choices)]
    return {
        "A": sorted(
            experiment_a,
            key=lambda row: (row["category"], row["prompt_id"]),
        ),
        "B": sorted(
            experiment_b,
            key=lambda row: (row["category"], row["prompt_id"]),
        ),
    }


def write_experiment_versions(
    *,
    output_dir: Path,
    experiment: str,
    selected: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """1実験について、表示位置を循環した3版を書く。"""
    version_orders_by_name = version_orders(len(selected), seed=seed)
    source_versions = ("A", "B", "C")
    manifest: dict[str, Any] = {}
    for counterbalance_version, source_version in zip(
        COUNTERBALANCE_VERSIONS,
        source_versions,
    ):
        version_dir = output_dir / f"counterbalance_version_{counterbalance_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        orders = version_orders_by_name[source_version]
        public_rows = [
            public_record(row, item_number=index, order=orders[index - 1])
            for index, row in enumerate(selected, start=1)
        ]
        private_rows = [
            {
                **private_record(
                    row,
                    item_number=index,
                    order=orders[index - 1],
                ),
                "experiment": experiment,
                "counterbalance_version": counterbalance_version,
            }
            for index, row in enumerate(selected, start=1)
        ]
        title = (
            f"相談支援応答の7段階評価・実験{experiment}"
            f"（表示順{counterbalance_version}）"
        )
        write_jsonl(version_dir / "form_items_public.jsonl", public_rows)
        write_jsonl(version_dir / "private_model_mapping.jsonl", private_rows)
        write_csv(version_dir / "google_form_items.csv", public_rows)
        write_markdown(
            version_dir / "google_form_sections.md",
            public_rows,
            title,
        )
        write_apps_script(
            version_dir / "create_google_form.gs",
            public_rows,
            title,
        )
        position_counts = Counter(
            (position, model)
            for row in private_rows
            for position, model in row["position_to_model"].items()
        )
        manifest[counterbalance_version] = {
            "count": len(public_rows),
            "position_counts": {
                f"{position}:{model}": count
                for (position, model), count in sorted(position_counts.items())
            },
            "public_jsonl_sha256": sha256_file(
                version_dir / "form_items_public.jsonl"
            ),
            "private_mapping_sha256": sha256_file(
                version_dir / "private_model_mapping.jsonl"
            ),
        }
    return manifest


def write_assignment_template(path: Path) -> None:
    """参加者の均等割当を記録する空のCSVを作る。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "participant_id",
                "assignment_group",
                "experiment",
                "counterbalance_version",
                "completed",
                "notes",
            ),
        )
        writer.writeheader()
        for experiment in EXPERIMENTS:
            for version in COUNTERBALANCE_VERSIONS:
                writer.writerow(
                    {
                        "participant_id": "",
                        "assignment_group": f"{experiment}{version}",
                        "experiment": experiment,
                        "counterbalance_version": version,
                        "completed": "",
                        "notes": "",
                    }
                )


def validate_split(
    *,
    selected: list[dict[str, Any]],
    experiments: dict[str, list[dict[str, Any]]],
) -> None:
    """分割の完全性と非重複を検証する。"""
    source_ids = {row["prompt_id"] for row in selected}
    ids_a = {row["prompt_id"] for row in experiments["A"]}
    ids_b = {row["prompt_id"] for row in experiments["B"]}
    if ids_a & ids_b:
        raise ValueError("実験A/Bに重複promptがあります。")
    if ids_a | ids_b != source_ids:
        raise ValueError("実験A/Bの和が元の20件と一致しません。")
    categories = {row["category"] for row in selected}
    for experiment in EXPERIMENTS:
        rows = experiments[experiment]
        counts = Counter(row["category"] for row in rows)
        if set(counts) != categories or any(count != 1 for count in counts.values()):
            raise ValueError(
                f"実験{experiment}がカテゴリ均衡ではありません: {counts}"
            )


def main() -> int:
    """2実験×3表示順のGoogle Form成果物を生成する。"""
    args = parse_args()
    response_path = args.v2_run / "three_model_responses.jsonl"
    axis_scores = load_axis_scores(
        v2_run=args.v2_run,
        topconf_run=args.topconf_run,
    )
    candidates = build_candidates(
        response_path=response_path,
        axis_scores=axis_scores,
    )
    selected = select_oracle_enriched(candidates, total=args.count)
    if len(selected) != 20:
        raise ValueError("実験A/Bは20件を10件ずつ分ける設計です。")
    experiments = split_category_pairs(selected)
    validate_split(selected=selected, experiments=experiments)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = selection_diagnostics(
        selected,
        permutations=args.diagnostic_permutations,
        seed=args.seed,
    )
    write_json(args.output_dir / "selection_conditioned_diagnostics.json", diagnostics)
    write_json(
        args.output_dir / "questionnaire_spec.json",
        {
            "version": "esconv_google_form_likert_blocked.v3",
            "experiments": 2,
            "items_per_participant": 10,
            "likert_ratings_per_participant": (
                10 * 3 * len(LIKERT_STATEMENTS)
            ),
            "final_choices_per_participant": 10,
            "likert_anchors": LIKERT_ANCHORS,
            "statements": list(LIKERT_STATEMENTS),
            "assignment_groups": [
                f"{experiment}{version}"
                for experiment in EXPERIMENTS
                for version in COUNTERBALANCE_VERSIONS
            ],
        },
    )

    experiment_manifests: dict[str, Any] = {}
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        rows = experiments[experiment]
        experiment_dir = args.output_dir / f"experiment_{experiment.lower()}"
        versions = write_experiment_versions(
            output_dir=experiment_dir,
            experiment=experiment,
            selected=rows,
            seed=args.seed + experiment_index,
        )
        experiment_manifests[experiment] = {
            "count": len(rows),
            "category_counts": dict(
                sorted(Counter(row["category"] for row in rows).items())
            ),
            "prompt_ids": [row["prompt_id"] for row in rows],
            "mean_basis_advantage": mean(
                row["basis_advantage_over_best_control"] for row in rows
            ),
            "counterbalance_versions": versions,
        }

    write_assignment_template(args.output_dir / "participant_assignment_template.csv")
    write_json(
        args.output_dir / "block_manifest.json",
        {
            "version": "esconv_oracle_enriched_likert_blocks.v3",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "source_selected_count": len(selected),
            "split_rule": (
                "各10カテゴリの2件を実験A/Bへ1件ずつ割り当て、"
                "BASiS Oracle優位度の合計差が最小となる組合せを選ぶ。"
            ),
            "posthoc_selection": True,
            "inference_scope": (
                "OracleでBASiS優位が確認された場面に限定した対象化ユーザ評価。"
            ),
            "experiments": experiment_manifests,
            "participant_assignment": (
                "参加者をA1/A2/A3/B1/B2/B3へできるだけ同数に割り当て、"
                "各参加者は1グループだけを評価する。"
            ),
            "source": {
                "responses": response_path.as_posix(),
                "responses_sha256": sha256_file(response_path),
                "v2_run": args.v2_run.as_posix(),
                "topconf_run": args.topconf_run.as_posix(),
            },
        },
    )
    print(
        "ESConv Likert評価を実験A/Bへ分割しました: "
        f"{args.output_dir} (A={len(experiments['A'])}, B={len(experiments['B'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
