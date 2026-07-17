"""BASiS vs Randomの人手A/B評価用itemを作成する。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ORACLE_RUN_DIR = Path(
    "artifacts/evaluations/oracle_eval_runs/"
    "esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/user_eval/items")
DEFAULT_SEED = 20260619
DEFAULT_TOTAL_ITEMS = 30
DEFAULT_BASIS_WIN_ITEMS = 20
DEFAULT_RANDOM_WIN_ITEMS = 5
DEFAULT_CLOSE_ITEMS = 5
DEFAULT_CLOSE_THRESHOLD = 3.0

SOURCE_BASIS = "basis"
SOURCE_RANDOM = "random"
STRATUM_BASIS_WIN = "oracle_basis_win"
STRATUM_RANDOM_WIN = "oracle_random_win"
STRATUM_CLOSE = "oracle_close"


@dataclass(frozen=True)
class SelectionConfig:
    """人手評価item選定の設定。"""

    seed: int = DEFAULT_SEED
    total_items: int = DEFAULT_TOTAL_ITEMS
    basis_win_items: int = DEFAULT_BASIS_WIN_ITEMS
    random_win_items: int = DEFAULT_RANDOM_WIN_ITEMS
    close_items: int = DEFAULT_CLOSE_ITEMS
    close_threshold: float = DEFAULT_CLOSE_THRESHOLD

    @property
    def stratum_targets(self) -> Counter[str]:
        """stratumごとの目標件数を返す。"""
        return Counter(
            {
                STRATUM_BASIS_WIN: self.basis_win_items,
                STRATUM_RANDOM_WIN: self.random_win_items,
                STRATUM_CLOSE: self.close_items,
            }
        )


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(
        description="Oracle評価済みBASiS/Random応答から人手A/B評価用itemを作成します。"
    )
    parser.add_argument("--oracle-run-dir", default=DEFAULT_ORACLE_RUN_DIR.as_posix())
    parser.add_argument("--responses", default=None)
    parser.add_argument("--judgments", default=None)
    parser.add_argument("--oracle-manifest", default=None)
    parser.add_argument("--prompts", default="configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--items-output", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--selection-csv", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--total-items", type=int, default=DEFAULT_TOTAL_ITEMS)
    parser.add_argument("--basis-win-items", type=int, default=DEFAULT_BASIS_WIN_ITEMS)
    parser.add_argument("--random-win-items", type=int, default=DEFAULT_RANDOM_WIN_ITEMS)
    parser.add_argument("--close-items", type=int, default=DEFAULT_CLOSE_ITEMS)
    parser.add_argument("--close-threshold", type=float, default=DEFAULT_CLOSE_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} をJSONとして読めません: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} がJSON objectではありません。")
            records.append(payload)
    if not records:
        raise ValueError(f"有効なレコードがありません: {path}")
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """JSONLを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    """JSONを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_key(record: dict[str, Any]) -> str:
    """prompt_id/item_idを返す。"""
    key = str(record.get("prompt_id") or record.get("item_id") or record.get("id") or "").strip()
    if not key:
        raise ValueError(f"prompt_idを取得できません: {record}")
    return key


def records_by_key(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """prompt_idでレコードを引ける辞書を作る。"""
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in result:
            raise ValueError(f"prompt_idが重複しています: {key}")
        result[key] = record
    return result


def basis_score(judgment: dict[str, Any]) -> float:
    """Oracle上のBASiSスコアを返す。"""
    return float(
        judgment.get("weighted_esconv_overall_score_base")
        or judgment.get("score_base")
        or judgment.get("base_score")
    )


def random_score(judgment: dict[str, Any]) -> float:
    """Oracle上のRandomスコアを返す。"""
    return float(
        judgment.get("weighted_esconv_overall_score_dpo")
        or judgment.get("score_dpo")
        or judgment.get("dpo_score")
    )


def oracle_stratum(judgment: dict[str, Any], *, close_threshold: float) -> str:
    """Oracleスコア差から選定stratumを付ける。"""
    gap = basis_score(judgment) - random_score(judgment)
    if abs(gap) < close_threshold:
        return STRATUM_CLOSE
    if gap > 0:
        return STRATUM_BASIS_WIN
    return STRATUM_RANDOM_WIN


def normalize_oracle_winner(winner: str) -> str:
    """既存runnerのbase/dpo表記をBASiS/Random表記へ直す。"""
    if winner == "base":
        return SOURCE_BASIS
    if winner == "dpo":
        return SOURCE_RANDOM
    return "tie"


def build_candidate_records(
    responses: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    *,
    close_threshold: float,
) -> list[dict[str, Any]]:
    """responsesとjudgmentsを結合し、人手評価候補へ整形する。"""
    response_by_id = records_by_key(responses)
    candidates: list[dict[str, Any]] = []
    for judgment in judgments:
        item_id = record_key(judgment)
        response = response_by_id.get(item_id)
        if response is None:
            raise ValueError(f"responses.jsonlに対応レコードがありません: {item_id}")
        basis_response = str(response.get("base_response") or "").strip()
        random_response = str(response.get("dpo_response") or "").strip()
        if not basis_response or not random_response:
            raise ValueError(f"BASiS/Random応答が空です: {item_id}")
        basis = basis_score(judgment)
        random_value = random_score(judgment)
        gap = basis - random_value
        stratum = oracle_stratum(judgment, close_threshold=close_threshold)
        candidates.append(
            {
                "item_id": item_id,
                "prompt": response.get("prompt") or judgment.get("prompt") or "",
                "history": response.get("history") or judgment.get("history") or [],
                "category": response.get("category") or judgment.get("category") or "",
                "axis_focus": response.get("axis_focus") or judgment.get("axis_focus") or [],
                "basis_response": basis_response,
                "random_response": random_response,
                "stratum": stratum,
                "oracle_winner": normalize_oracle_winner(str(judgment.get("winner") or "")),
                "oracle_winner_raw": judgment.get("winner"),
                "basis_score": basis,
                "random_score": random_value,
                "score_gap": gap,
            }
        )
    return candidates


def category_targets(categories: list[str], total_items: int) -> dict[str, int]:
    """カテゴリ件数ができるだけ均等になる目標値を返す。"""
    if not categories:
        return {}
    base = total_items // len(categories)
    remainder = total_items % len(categories)
    targets: dict[str, int] = {}
    for index, category in enumerate(sorted(categories)):
        targets[category] = base + (1 if index < remainder else 0)
    return targets


def _combo_order_key(
    combo: tuple[dict[str, Any], ...],
    *,
    rng: random.Random,
) -> tuple[int, float, float]:
    """DFSで使うカテゴリ内組み合わせの順序を決める。"""
    strata = {str(record["stratum"]) for record in combo}
    mean_abs_gap = sum(abs(float(record["score_gap"])) for record in combo) / len(combo)
    return (len(strata), mean_abs_gap, rng.random())


def select_exact_balanced(
    candidates: list[dict[str, Any]],
    *,
    config: SelectionConfig,
    targets_by_category: dict[str, int],
) -> list[dict[str, Any]] | None:
    """カテゴリ目標とstratum目標を同時に満たす選定を探す。"""
    rng = random.Random(config.seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_category[str(record["category"])].append(record)

    category_names = sorted(targets_by_category)
    combos_by_category: dict[str, list[tuple[tuple[dict[str, Any], ...], Counter[str]]]] = {}
    for category in category_names:
        count = targets_by_category[category]
        category_records = by_category.get(category, [])
        if len(category_records) < count:
            return None
        combos = list(itertools.combinations(category_records, count))
        combos.sort(key=lambda combo: _combo_order_key(combo, rng=rng), reverse=True)
        combos_by_category[category] = [
            (combo, Counter(str(record["stratum"]) for record in combo)) for combo in combos
        ]

    target_counts = config.stratum_targets

    def can_still_reach(index: int, current: Counter[str]) -> bool:
        remaining_categories = category_names[index:]
        possible: Counter[str] = Counter()
        for category in remaining_categories:
            need_count = targets_by_category[category]
            category_records = by_category.get(category, [])
            for stratum in target_counts:
                possible[stratum] += min(
                    need_count,
                    sum(1 for record in category_records if record["stratum"] == stratum),
                )
        return all(current[stratum] + possible[stratum] >= target_counts[stratum] for stratum in target_counts)

    def dfs(
        index: int,
        current_counts: Counter[str],
        selected: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if index >= len(category_names):
            if len(selected) == config.total_items and current_counts == target_counts:
                return selected
            return None
        category = category_names[index]
        for combo, combo_counts in combos_by_category[category]:
            next_counts = current_counts + combo_counts
            if any(next_counts[stratum] > target_counts[stratum] for stratum in target_counts):
                continue
            if not can_still_reach(index + 1, next_counts):
                continue
            result = dfs(index + 1, next_counts, selected + list(combo))
            if result is not None:
                return result
        return None

    return dfs(0, Counter(), [])


def select_greedy_fallback(
    candidates: list[dict[str, Any]],
    *,
    config: SelectionConfig,
) -> list[dict[str, Any]]:
    """完全制約が解けない場合にstratum目標を優先して近似選定する。"""
    rng = random.Random(config.seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    category_goal = max(1, config.total_items // max(1, len({record["category"] for record in candidates})))

    while len(selected) < config.total_items:
        best_index: int | None = None
        best_score: tuple[int, int, float] | None = None
        for index, record in enumerate(shuffled):
            stratum = str(record["stratum"])
            if stratum_counts[stratum] >= config.stratum_targets[stratum]:
                continue
            category = str(record["category"])
            category_under = 1 if category_counts[category] < category_goal else 0
            remaining_need = config.stratum_targets[stratum] - stratum_counts[stratum]
            score = (category_under, remaining_need, rng.random())
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        record = shuffled.pop(best_index)
        selected.append(record)
        stratum_counts[str(record["stratum"])] += 1
        category_counts[str(record["category"])] += 1
    return selected


def select_user_eval_records(
    candidates: list[dict[str, Any]],
    *,
    config: SelectionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """人手評価に使うレコードを選ぶ。"""
    if config.total_items != config.basis_win_items + config.random_win_items + config.close_items:
        raise ValueError("total-itemsとstratum別件数の合計が一致していません。")

    available = Counter(str(record["stratum"]) for record in candidates)
    for stratum, required in config.stratum_targets.items():
        if available[stratum] < required:
            raise ValueError(f"{stratum} が不足しています: required={required}, available={available[stratum]}")

    categories = sorted({str(record["category"]) for record in candidates})
    targets_by_category = category_targets(categories, config.total_items)
    selected = select_exact_balanced(candidates, config=config, targets_by_category=targets_by_category)
    constraints_relaxed = False
    approximation_note = ""
    if selected is None:
        constraints_relaxed = True
        approximation_note = (
            "カテゴリ均等制約とstratum件数を同時に満たす解がないため、"
            "stratum件数を優先するgreedy fallbackで近似選定した。"
        )
        selected = select_greedy_fallback(candidates, config=config)

    if len(selected) != config.total_items:
        raise RuntimeError(f"選定件数が不足しました: selected={len(selected)}, target={config.total_items}")

    selected = assign_display_order(selected, seed=config.seed)
    info = {
        "constraints_relaxed": constraints_relaxed,
        "approximation_note": approximation_note,
        "category_targets": targets_by_category,
        "available_strata": dict(available),
    }
    return selected, info


def assign_display_order(records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """A/B表示順とitem提示順をseed固定で付与する。"""
    rng = random.Random(seed + 7919)
    shuffled = list(records)
    rng.shuffle(shuffled)
    basis_a_count = len(shuffled) // 2
    basis_a_indices = set(rng.sample(range(len(shuffled)), basis_a_count))
    output: list[dict[str, Any]] = []
    for display_index, record in enumerate(shuffled, start=1):
        item = dict(record)
        if display_index - 1 in basis_a_indices:
            item["model_a_response"] = item["basis_response"]
            item["model_b_response"] = item["random_response"]
            item["model_a_source"] = SOURCE_BASIS
            item["model_b_source"] = SOURCE_RANDOM
            item["basis_position"] = "A"
            item["random_position"] = "B"
            item["displayed_order"] = "basis_a_random_b"
        else:
            item["model_a_response"] = item["random_response"]
            item["model_b_response"] = item["basis_response"]
            item["model_a_source"] = SOURCE_RANDOM
            item["model_b_source"] = SOURCE_BASIS
            item["basis_position"] = "B"
            item["random_position"] = "A"
            item["displayed_order"] = "random_a_basis_b"
        item["display_index"] = display_index
        output.append(item)
    return output


def build_eval_item(record: dict[str, Any], *, source_paths: dict[str, str]) -> dict[str, Any]:
    """GUIが読む人手評価itemへ変換する。"""
    return {
        "item_id": record["item_id"],
        "display_index": record["display_index"],
        "category": record["category"],
        "stratum": record["stratum"],
        "prompt": record["prompt"],
        "history": record["history"],
        "axis_focus": record["axis_focus"],
        "model_a_response": record["model_a_response"],
        "model_b_response": record["model_b_response"],
        "model_a_source": record["model_a_source"],
        "model_b_source": record["model_b_source"],
        "displayed_order": record["displayed_order"],
        "basis_position": record["basis_position"],
        "random_position": record["random_position"],
        "oracle_winner": record["oracle_winner"],
        "oracle_winner_raw": record["oracle_winner_raw"],
        "basis_score": round(float(record["basis_score"]), 6),
        "random_score": round(float(record["random_score"]), 6),
        "score_gap": round(float(record["score_gap"]), 6),
        "source_paths": source_paths,
    }


def selection_row(record: dict[str, Any], *, source_paths: dict[str, str]) -> dict[str, Any]:
    """選定CSV/manifest用の1行を作る。"""
    return {
        "display_index": record["display_index"],
        "item_id": record["item_id"],
        "category": record["category"],
        "stratum": record["stratum"],
        "oracle_winner": record["oracle_winner"],
        "oracle_winner_raw": record["oracle_winner_raw"],
        "basis_score": round(float(record["basis_score"]), 6),
        "random_score": round(float(record["random_score"]), 6),
        "score_gap": round(float(record["score_gap"]), 6),
        "basis_position": record["basis_position"],
        "random_position": record["random_position"],
        "displayed_order": record["displayed_order"],
        "responses_path": source_paths["responses_path"],
        "judgments_path": source_paths["judgments_path"],
        "oracle_manifest_path": source_paths["oracle_manifest_path"],
    }


def write_selection_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """選定内容をCSVに保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "display_index",
        "item_id",
        "category",
        "stratum",
        "oracle_winner",
        "oracle_winner_raw",
        "basis_score",
        "random_score",
        "score_gap",
        "basis_position",
        "random_position",
        "displayed_order",
        "responses_path",
        "judgments_path",
        "oracle_manifest_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    *,
    config: SelectionConfig,
    selected_rows: list[dict[str, Any]],
    selection_info: dict[str, Any],
    source_paths: dict[str, str],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """再現性情報を含むmanifestを作る。"""
    category_counts = Counter(row["category"] for row in selected_rows)
    stratum_counts = Counter(row["stratum"] for row in selected_rows)
    oracle_winner_counts = Counter(row["oracle_winner"] for row in selected_rows)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Oracle評価を補足する研究室内のBASiS vs Random人手A/B評価",
        "selection_method": (
            "Oracle weighted_esconv_overallのBASiS-Random差で層化し、"
            "BASiS優位20件、Random優位5件、小差5件をseed固定で選定。"
            "カテゴリは可能な範囲で均等化する。"
        ),
        "representativeness_note": (
            "この30件はBASiSの有効性が人手評価でも確認できるかを見る補足評価用であり、"
            "完全無作為な代表サンプルではない。"
        ),
        "seed": config.seed,
        "close_threshold": config.close_threshold,
        "target_counts": dict(config.stratum_targets),
        "actual_counts": {
            "total": len(selected_rows),
            "strata": dict(stratum_counts),
            "categories": dict(category_counts),
            "oracle_winners": dict(oracle_winner_counts),
        },
        "constraints_relaxed": selection_info["constraints_relaxed"],
        "approximation_note": selection_info["approximation_note"],
        "category_targets": selection_info["category_targets"],
        "available_strata": selection_info["available_strata"],
        "source_paths": source_paths,
        "output_paths": output_paths,
        "basis_definition": "responses.jsonlのbase_response/base fieldをBASiS/Bayes-DPOとして扱う。",
        "random_definition": "responses.jsonlのdpo_response/dpo fieldをRandom-DPOとして扱う。",
        "selected_items": selected_rows,
    }


def resolve_paths(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Path]]:
    """入力・出力パスを確定する。"""
    oracle_run_dir = Path(args.oracle_run_dir)
    responses = Path(args.responses) if args.responses else oracle_run_dir / "responses.jsonl"
    judgments = Path(args.judgments) if args.judgments else oracle_run_dir / "judgments.jsonl"
    oracle_manifest = Path(args.oracle_manifest) if args.oracle_manifest else oracle_run_dir / "manifest.json"
    output_dir = Path(args.output_dir)
    items_output = Path(args.items_output) if args.items_output else output_dir / "user_eval_items.jsonl"
    manifest_output = (
        Path(args.manifest_output) if args.manifest_output else output_dir / "selection_manifest.json"
    )
    selection_csv = Path(args.selection_csv) if args.selection_csv else output_dir / "selected_items.csv"
    return (
        {
            "responses": responses,
            "judgments": judgments,
            "oracle_manifest": oracle_manifest,
            "prompts": Path(args.prompts),
        },
        {
            "items_output": items_output,
            "manifest_output": manifest_output,
            "selection_csv": selection_csv,
        },
    )


def main() -> None:
    """CLI entrypoint。"""
    args = parse_args()
    input_paths, output_paths = resolve_paths(args)
    config = SelectionConfig(
        seed=args.seed,
        total_items=args.total_items,
        basis_win_items=args.basis_win_items,
        random_win_items=args.random_win_items,
        close_items=args.close_items,
        close_threshold=args.close_threshold,
    )

    for label, path in input_paths.items():
        if label == "prompts":
            continue
        if not path.exists():
            raise FileNotFoundError(f"{label} が見つかりません: {path}")

    responses = read_jsonl(input_paths["responses"])
    judgments = read_jsonl(input_paths["judgments"])
    candidates = build_candidate_records(
        responses,
        judgments,
        close_threshold=config.close_threshold,
    )
    selected, selection_info = select_user_eval_records(candidates, config=config)
    source_paths = {
        "responses_path": input_paths["responses"].as_posix(),
        "judgments_path": input_paths["judgments"].as_posix(),
        "oracle_manifest_path": input_paths["oracle_manifest"].as_posix(),
        "prompts_path": input_paths["prompts"].as_posix(),
    }
    eval_items = [build_eval_item(record, source_paths=source_paths) for record in selected]
    selected_rows = [selection_row(record, source_paths=source_paths) for record in selected]
    manifest = build_manifest(
        config=config,
        selected_rows=selected_rows,
        selection_info=selection_info,
        source_paths=source_paths,
        output_paths={key: path.as_posix() for key, path in output_paths.items()},
    )

    print("User evaluation item selection")
    print(f"  seed: {config.seed}")
    print(f"  selected: {len(selected_rows)}")
    print(f"  strata: {dict(Counter(row['stratum'] for row in selected_rows))}")
    print(f"  categories: {dict(Counter(row['category'] for row in selected_rows))}")
    print(f"  constraints_relaxed: {selection_info['constraints_relaxed']}")

    if args.dry_run:
        for row in selected_rows:
            print(
                "  {item_id} {category} {stratum} gap={score_gap}".format(
                    **row,
                )
            )
        return

    write_jsonl(eval_items, output_paths["items_output"])
    write_json(manifest, output_paths["manifest_output"])
    write_selection_csv(selected_rows, output_paths["selection_csv"])
    print(f"評価itemを書き出しました: {output_paths['items_output']}")
    print(f"選定manifestを書き出しました: {output_paths['manifest_output']}")
    print(f"選定CSVを書き出しました: {output_paths['selection_csv']}")


if __name__ == "__main__":
    main()
