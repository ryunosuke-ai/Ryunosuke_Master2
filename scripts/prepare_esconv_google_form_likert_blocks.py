#!/usr/bin/env python3
"""ESConv Likert評価20件を、カテゴリ均衡した2実験へ分割する。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
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
    sha256_file,
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
    "artifacts/user_eval/google_forms/esconv_human_reviewed_likert_two_forms_v6"
)
DEFAULT_SELECTION_CONFIG = Path(
    "configs/evaluations/esconv_user_eval_human_review_v1.json"
)
EXPERIMENTS = ("A", "B")


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="ESConv Likert評価20件を10件ずつの実験A/Bへ分割します。"
    )
    parser.add_argument("--v2-run", type=Path, default=DEFAULT_V2_RUN)
    parser.add_argument("--topconf-run", type=Path, default=DEFAULT_TOPCONF_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=DEFAULT_SELECTION_CONFIG,
    )
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


def select_discriminative_items(
    candidates: list[dict[str, Any]],
    *,
    total: int,
) -> list[dict[str, Any]]:
    """BASiSが高く、最良controlとの差が大きいitemを選ぶ。"""
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["basis_advantage_over_best_control"],
            row["representative_means"]["basis"],
            row["prompt_id"],
        ),
        reverse=True,
    )
    selected = ranked[:total]
    if len(selected) != total:
        raise ValueError(f"識別力重視itemが不足しています: {len(selected)}/{total}")
    if any(row["representative_means"]["basis"] < 8.5 for row in selected):
        raise ValueError("選定itemにBASiS代表5軸平均8.5未満が含まれます。")
    if any(row["basis_advantage_over_best_control"] < 0.5 for row in selected):
        raise ValueError("選定itemに最良controlとの差0.5未満が含まれます。")
    return sorted(
        selected,
        key=lambda row: (
            -row["basis_advantage_over_best_control"],
            row["prompt_id"],
        ),
    )


def select_human_reviewed_items(
    candidates: list[dict[str, Any]],
    *,
    config_path: Path,
    total: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """定性的監査で固定したprompt IDを再現する。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    items = config.get("items")
    if not isinstance(items, list):
        raise ValueError("人手可読性監査configにitemsがありません。")
    prompt_ids = [str(item.get("prompt_id") or "") for item in items]
    if len(prompt_ids) != total or len(set(prompt_ids)) != total:
        raise ValueError(
            f"人手可読性監査itemが不足または重複しています: "
            f"{len(prompt_ids)}/{total}"
        )
    by_id = {row["prompt_id"]: row for row in candidates}
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
    if missing:
        raise ValueError(f"人手可読性監査itemが候補にありません: {missing}")
    selected = [by_id[prompt_id] for prompt_id in prompt_ids]
    if any(row["representative_means"]["basis"] < 8.0 for row in selected):
        raise ValueError("人手可読性監査itemにBASiS平均8.0未満が含まれます。")
    return selected, config


def split_discriminative_items(
    selected: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """20件をカテゴリ構成とOracle優位度が近い10件ずつへ分ける。"""
    if len(selected) != 20:
        raise ValueError("識別力重視分割は20件を前提とします。")
    categories = sorted({row["category"] for row in selected})
    best: tuple[float, float, tuple[int, ...]] | None = None
    for indices in itertools.combinations(range(len(selected)), 10):
        index_set = set(indices)
        rows_a = [selected[index] for index in indices]
        rows_b = [
            row for index, row in enumerate(selected) if index not in index_set
        ]
        counts_a = Counter(row["category"] for row in rows_a)
        counts_b = Counter(row["category"] for row in rows_b)
        category_imbalance = sum(
            abs(counts_a[category] - counts_b[category])
            for category in categories
        )
        advantage_imbalance = abs(
            sum(row["basis_advantage_over_best_control"] for row in rows_a)
            - sum(row["basis_advantage_over_best_control"] for row in rows_b)
        )
        candidate = (category_imbalance, advantage_imbalance, indices)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("識別力重視itemを分割できませんでした。")
    indices_a = set(best[2])
    return {
        "A": sorted(
            [
                row
                for index, row in enumerate(selected)
                if index in indices_a
            ],
            key=lambda row: (row["category"], row["prompt_id"]),
        ),
        "B": sorted(
            [
                row
                for index, row in enumerate(selected)
                if index not in indices_a
            ],
            key=lambda row: (row["category"], row["prompt_id"]),
        ),
    }


def balanced_single_form_orders(
    count: int,
    *,
    seed: int,
) -> list[tuple[str, str, str]]:
    """1フォーム内で各モデルのA/B/C位置を3〜4回に均衡する。"""
    base = list(MODEL_KEYS)
    rotation = seed % len(base)
    base = base[rotation:] + base[:rotation]
    latin_orders = [
        tuple(base[index:] + base[:index])
        for index in range(len(base))
    ]
    orders = [
        latin_orders[index % len(latin_orders)]
        for index in range(count)
    ]
    # item順との規則的な対応を避けつつ、位置ごとの件数は維持する。
    random.Random(seed).shuffle(orders)
    return orders


def write_experiment(
    *,
    output_dir: Path,
    experiment: str,
    selected: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """1実験につき、位置を均衡した1つのフォームを書く。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    orders = balanced_single_form_orders(len(selected), seed=seed)
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
        }
        for index, row in enumerate(selected, start=1)
    ]
    title = f"相談支援応答の7段階評価・実験{experiment}"
    write_jsonl(output_dir / "form_items_public.jsonl", public_rows)
    write_jsonl(output_dir / "private_model_mapping.jsonl", private_rows)
    write_csv(output_dir / "google_form_items.csv", public_rows)
    write_markdown(
        output_dir / "google_form_sections.md",
        public_rows,
        title,
    )
    write_apps_script(
        output_dir / "create_google_form.gs",
        public_rows,
        title,
    )
    position_counts = Counter(
        (position, model)
        for row in private_rows
        for position, model in row["position_to_model"].items()
    )
    return {
        "count": len(public_rows),
        "position_counts": {
            f"{position}:{model}": count
            for (position, model), count in sorted(position_counts.items())
        },
        "public_jsonl_sha256": sha256_file(
            output_dir / "form_items_public.jsonl"
        ),
        "private_mapping_sha256": sha256_file(
            output_dir / "private_model_mapping.jsonl"
        ),
    }


def write_assignment_template(path: Path) -> None:
    """参加者の均等割当を記録する空のCSVを作る。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "participant_name",
                "assignment_group",
                "experiment",
                "completed",
                "notes",
            ),
        )
        writer.writeheader()
        for experiment in EXPERIMENTS:
            writer.writerow(
                {
                    "participant_name": "",
                    "assignment_group": experiment,
                    "experiment": experiment,
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
    """実験A/Bの2つのGoogle Form成果物を生成する。"""
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
    selected, review_config = select_human_reviewed_items(
        candidates,
        config_path=args.selection_config,
        total=args.count,
    )
    if len(selected) != 20:
        raise ValueError("実験A/Bは20件を10件ずつ分ける設計です。")
    experiments = split_discriminative_items(selected)
    source_ids = {row["prompt_id"] for row in selected}
    ids_a = {row["prompt_id"] for row in experiments["A"]}
    ids_b = {row["prompt_id"] for row in experiments["B"]}
    if ids_a & ids_b or ids_a | ids_b != source_ids:
        raise ValueError("実験A/Bの重複または欠落を検出しました。")

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
            "version": "esconv_google_form_human_reviewed_two_forms.v6",
            "experiments": 2,
            "forms": 2,
            "items_per_participant": 10,
            "participant_field": "full_name",
            "contains_personal_data": True,
            "likert_ratings_per_participant": (
                10 * 3 * len(LIKERT_STATEMENTS)
            ),
            "final_choices_per_participant": 10,
            "likert_anchors": LIKERT_ANCHORS,
            "statements": list(LIKERT_STATEMENTS),
            "assignment_groups": list(EXPERIMENTS),
            "position_control": (
                "各フォーム内で各モデルが応答A/B/Cの各位置へ"
                "3回または4回現れるよう固定配置する。"
            ),
        },
    )

    experiment_manifests: dict[str, Any] = {}
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        rows = experiments[experiment]
        experiment_dir = args.output_dir / f"experiment_{experiment.lower()}"
        form_manifest = write_experiment(
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
            "form": form_manifest,
        }

    write_assignment_template(args.output_dir / "participant_assignment_template.csv")
    write_json(
        args.output_dir / "block_manifest.json",
        {
            "version": "esconv_human_reviewed_likert_two_forms.v6",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "source_selected_count": len(selected),
            "split_rule": (
                "Oracle候補をLLMが人間の可読性・支援スタイル対比の明瞭さで"
                "全件監査して固定した20件を使い、カテゴリ構成差を最小化した"
                "上で、Oracle優位度の合計差が最小となる10件ずつへ分割する。"
            ),
            "selection_rule": (
                "感情受容、非指示性、助言タイミング、比較応答との差、"
                "BASiS応答自体の自然さを定性的に確認した固定configを使う。"
            ),
            "selection_config": args.selection_config.as_posix(),
            "selection_config_sha256": sha256_file(args.selection_config),
            "qualitative_contrast_counts": dict(
                sorted(
                    Counter(
                        str(item.get("contrast") or "unknown")
                        for item in review_config["items"]
                    ).items()
                )
            ),
            "selected_category_counts": dict(
                sorted(Counter(row["category"] for row in selected).items())
            ),
            "minimum_basis_representative_mean": min(
                row["representative_means"]["basis"] for row in selected
            ),
            "minimum_basis_advantage": min(
                row["basis_advantage_over_best_control"] for row in selected
            ),
            "mean_basis_advantage": mean(
                row["basis_advantage_over_best_control"] for row in selected
            ),
            "posthoc_selection": True,
            "inference_scope": (
                "OracleでBASiS優位が確認された場面に限定した対象化ユーザ評価。"
            ),
            "experiments": experiment_manifests,
            "participant_assignment": (
                "参加者を実験A/Bへできるだけ同数に割り当て、"
                "各参加者は一方だけを評価する。氏名は個人情報として"
                "研究担当者だけが取り扱う。"
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
