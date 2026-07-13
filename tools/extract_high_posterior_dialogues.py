"""ベイズスコア済み対話から高posterior応答を抽出する。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.transition_bayes_model import TransitionBayesModel, load_transition_bayes_model
from tools.jsonl_utils import read_jsonl_records


DEFAULT_INPUT_PATH = "artifacts/scored_dialogues/dailydialog_transition_scored.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/datasets/dailydialog_selected_en.jsonl"
DEFAULT_MIN_POSTERIOR = 0.75
DEFAULT_PREFERRED_STATES = (
    "opening_invitation",
    "setting_sensory_detail",
    "activity_social_detail",
)
DEFAULT_PREFERRED_OBSERVATIONS = (
    "ack_open_probe",
    "sensory_setting_focus",
    "activity_social_focus",
)
DEFAULT_LOW_PRIORITY_STATES = ("warm_closure",)
DEFAULT_EXCLUDED_STATES = ("off_style",)
DEFAULT_EXCLUDED_OBSERVATIONS = ("generic_or_unrelated",)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="高posteriorの文脈付き応答を抽出します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力スコア済みJSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--bayes-model", default="", help="指定時はpositive/negative statesから優先・除外ラベルを自動導出します。")
    parser.add_argument("--min-posterior", type=float, default=DEFAULT_MIN_POSTERIOR, help="抽出するposteriorの下限。")
    parser.add_argument("--min-context-turns", type=int, default=0, help="promptに含まれる文脈ターン数の下限。")
    parser.add_argument("--max-records", type=int, default=None, help="出力件数の上限。")
    parser.add_argument("--target-records", type=int, default=None, help="目標出力件数。指定時はmax-recordsより優先します。")
    parser.add_argument("--per-dialogue-limit", type=int, default=None, help="同一会話から採用する最大件数。")
    parser.add_argument("--prefer-states", default=",".join(DEFAULT_PREFERRED_STATES), help="優先する状態ラベルのカンマ区切り。")
    parser.add_argument("--prefer-observations", default=",".join(DEFAULT_PREFERRED_OBSERVATIONS), help="優先する観測ラベルのカンマ区切り。")
    parser.add_argument("--low-priority-states", default=",".join(DEFAULT_LOW_PRIORITY_STATES), help="減点する状態ラベルのカンマ区切り。")
    parser.add_argument("--exclude-states", default=",".join(DEFAULT_EXCLUDED_STATES), help="除外する状態ラベルのカンマ区切り。")
    parser.add_argument("--exclude-observations", default=",".join(DEFAULT_EXCLUDED_OBSERVATIONS), help="除外する観測ラベルのカンマ区切り。")
    parser.add_argument("--require-preferred", action="store_true", help="優先状態または優先観測に入る候補だけを採用します。")
    parser.add_argument("--sort-by-posterior", action="store_true", help="posterior降順で抽出します。")
    parser.add_argument("--sort-by-selection", action="store_true", help="selection_score降順で抽出します。")
    parser.add_argument("--dry-run", action="store_true", help="書き出さず、抽出件数だけ表示します。")
    return parser.parse_args()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    records, skipped = read_jsonl_records(
        path,
        missing_ok=False,
        strict=False,
        label="スコア済み入力JSONL",
    )
    if skipped:
        print(f"[WARN] スコア済み入力JSONLの壊れた行をskipしました: skipped={skipped}", flush=True)
    return [record for record in records if isinstance(record, dict)]


def _posterior(record: dict[str, Any]) -> float:
    """posteriorを数値として読む。"""
    value = record.get("posterior")
    if not isinstance(value, (int, float)):
        raise ValueError("`posterior` が数値でないレコードがあります。")
    return float(value)


def _label_set(value: str | tuple[str, ...] | list[str] | None) -> set[str]:
    """カンマ区切りまたは配列のラベル指定をsetへ変換する。"""
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def _label_text(value: set[str]) -> str:
    """ラベル集合をCLI互換のカンマ区切り文字列へ変換する。"""
    return ",".join(sorted(value))


def derive_selection_label_diagnostics(
    model: TransitionBayesModel,
    *,
    method: str = "mean_difference",
    minimum_margin: float = 0.0,
) -> dict[str, Any]:
    """モデルから選別ラベルとemission差の監査情報を導出する。"""
    if method not in {"mean_difference", "state_specific_margin"}:
        raise ValueError(f"未知の選別ラベル導出方式です: {method}")
    positive_states = set(model.positive_states)
    negative_states = set(model.negative_states)
    observation_details: dict[str, dict[str, Any]] = {}
    for observation in model.observations:
        positive_mean = sum(model.emission_likelihoods[state][observation] for state in positive_states) / max(1, len(positive_states))
        negative_mean = sum(model.emission_likelihoods[state][observation] for state in negative_states) / max(1, len(negative_states))
        positive_max = max(model.emission_likelihoods[state][observation] for state in positive_states)
        negative_max = max(model.emission_likelihoods[state][observation] for state in negative_states)
        score = (
            positive_mean - negative_mean
            if method == "mean_difference"
            else positive_max - negative_max
        )
        is_preferred = score > 0.0 if method == "mean_difference" else score >= minimum_margin
        is_excluded = score < 0.0 if method == "mean_difference" else score <= -minimum_margin
        classification = "neutral"
        if is_preferred:
            classification = "preferred"
        elif is_excluded:
            classification = "excluded"
        observation_details[observation] = {
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "positive_max": positive_max,
            "negative_max": negative_max,
            "margin": score,
            "classification": classification,
        }

    observation_scores = {
        observation: float(detail["margin"])
        for observation, detail in observation_details.items()
    }

    preferred_observations = {
        observation
        for observation, detail in observation_details.items()
        if detail["classification"] == "preferred"
    }
    if method == "mean_difference" and len(preferred_observations) > 4:
        preferred_observations = {
            observation
            for observation, _ in sorted(observation_scores.items(), key=lambda item: item[1], reverse=True)[:4]
        }

    excluded_observations = {
        observation
        for observation, detail in observation_details.items()
        if detail["classification"] == "excluded"
    }
    if method == "mean_difference" and len(excluded_observations) > 4:
        excluded_observations = {
            observation
            for observation, _ in sorted(observation_scores.items(), key=lambda item: item[1])[:4]
        }

    labels = {
        "prefer_states": _label_text(positive_states),
        "prefer_observations": _label_text(preferred_observations),
        "low_priority_states": "",
        "exclude_states": _label_text(negative_states),
        "exclude_observations": _label_text(excluded_observations),
    }
    return {
        "method": method,
        "minimum_margin": minimum_margin,
        "labels": labels,
        "observations": observation_details,
    }


def derive_selection_labels_from_model(
    model: TransitionBayesModel,
    *,
    method: str = "mean_difference",
    minimum_margin: float = 0.0,
) -> dict[str, str]:
    """状態遷移ベイズモデルから抽出優先・除外ラベルを導出する。"""
    return derive_selection_label_diagnostics(
        model,
        method=method,
        minimum_margin=minimum_margin,
    )["labels"]


def apply_bayes_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """bayes-model指定時、未変更の既定ラベルだけをモデル由来値で置換する。"""
    if not args.bayes_model:
        return args
    labels = derive_selection_labels_from_model(load_transition_bayes_model(args.bayes_model))
    if args.prefer_states == ",".join(DEFAULT_PREFERRED_STATES):
        args.prefer_states = labels["prefer_states"]
    if args.prefer_observations == ",".join(DEFAULT_PREFERRED_OBSERVATIONS):
        args.prefer_observations = labels["prefer_observations"]
    if args.low_priority_states == ",".join(DEFAULT_LOW_PRIORITY_STATES):
        args.low_priority_states = labels["low_priority_states"]
    if args.exclude_states == ",".join(DEFAULT_EXCLUDED_STATES):
        args.exclude_states = labels["exclude_states"]
    if args.exclude_observations == ",".join(DEFAULT_EXCLUDED_OBSERVATIONS):
        args.exclude_observations = labels["exclude_observations"]
    return args


def _context_turns(record: dict[str, Any]) -> int:
    """metadataから文脈ターン数を読む。"""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    value = metadata.get("context_turns")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _selection_score(
    record: dict[str, Any],
    *,
    prefer_states: set[str],
    prefer_observations: set[str],
    low_priority_states: set[str],
) -> float:
    """回想支援DPOに向いた候補ほど高い選別スコアを返す。"""
    posterior = _posterior(record)
    state = str(record.get("most_likely_state", ""))
    observation = str(record.get("observation", ""))
    score = posterior
    if state in prefer_states:
        score += 0.20
    if observation in prefer_observations:
        score += 0.25
    if state in low_priority_states:
        score -= 0.20
    observation_score = record.get("observation_score")
    if isinstance(observation_score, (int, float)):
        score += min(0.10, max(0.0, float(observation_score)) * 0.05)
    return score


def _selection_reason(
    record: dict[str, Any],
    *,
    prefer_states: set[str],
    prefer_observations: set[str],
    low_priority_states: set[str],
) -> str:
    """選別理由を短く作る。"""
    reasons = [f"posterior={_posterior(record):.3f}"]
    state = str(record.get("most_likely_state", ""))
    observation = str(record.get("observation", ""))
    if state in prefer_states:
        reasons.append(f"preferred_state={state}")
    elif state in low_priority_states:
        reasons.append(f"low_priority_state={state}")
    elif state:
        reasons.append(f"state={state}")
    if observation in prefer_observations:
        reasons.append(f"preferred_observation={observation}")
    elif observation:
        reasons.append(f"observation={observation}")
    context_turns = _context_turns(record)
    if context_turns:
        reasons.append(f"context_turns={context_turns}")
    return "; ".join(reasons)


def _with_selection_metadata(
    record: dict[str, Any],
    *,
    prefer_states: set[str],
    prefer_observations: set[str],
    low_priority_states: set[str],
) -> dict[str, Any]:
    """抽出結果に選別スコアと理由を付与する。"""
    selected = dict(record)
    selected["selection_score"] = _selection_score(
        record,
        prefer_states=prefer_states,
        prefer_observations=prefer_observations,
        low_priority_states=low_priority_states,
    )
    selected["selection_reason"] = _selection_reason(
        record,
        prefer_states=prefer_states,
        prefer_observations=prefer_observations,
        low_priority_states=low_priority_states,
    )
    return selected


def select_high_posterior_records(
    records: list[dict[str, Any]],
    *,
    min_posterior: float,
    max_records: int | None,
    sort_by_posterior: bool,
    min_context_turns: int = 0,
    target_records: int | None = None,
    per_dialogue_limit: int | None = None,
    prefer_states: str | tuple[str, ...] | list[str] | None = DEFAULT_PREFERRED_STATES,
    prefer_observations: str | tuple[str, ...] | list[str] | None = DEFAULT_PREFERRED_OBSERVATIONS,
    low_priority_states: str | tuple[str, ...] | list[str] | None = DEFAULT_LOW_PRIORITY_STATES,
    exclude_states: str | tuple[str, ...] | list[str] | None = DEFAULT_EXCLUDED_STATES,
    exclude_observations: str | tuple[str, ...] | list[str] | None = DEFAULT_EXCLUDED_OBSERVATIONS,
    require_preferred: bool = False,
    sort_by_selection: bool = False,
) -> list[dict[str, Any]]:
    """高posteriorかつ回想支援DPOに向いたレコードを抽出する。"""
    prefer_state_set = _label_set(prefer_states)
    prefer_observation_set = _label_set(prefer_observations)
    low_priority_state_set = _label_set(low_priority_states)
    exclude_state_set = _label_set(exclude_states)
    exclude_observation_set = _label_set(exclude_observations)

    selected = []
    for record in records:
        if _posterior(record) < min_posterior:
            continue
        if _context_turns(record) < min_context_turns:
            continue
        state = str(record.get("most_likely_state", ""))
        observation = str(record.get("observation", ""))
        if state in exclude_state_set or observation in exclude_observation_set:
            continue
        if require_preferred and state not in prefer_state_set and observation not in prefer_observation_set:
            continue
        selected.append(
            _with_selection_metadata(
                record,
                prefer_states=prefer_state_set,
                prefer_observations=prefer_observation_set,
                low_priority_states=low_priority_state_set,
            )
        )

    if sort_by_selection:
        selected = sorted(selected, key=lambda record: float(record["selection_score"]), reverse=True)
    if sort_by_posterior:
        selected = sorted(selected, key=_posterior, reverse=True)

    limit = target_records if target_records is not None else max_records
    if per_dialogue_limit is not None:
        limited = []
        counts: dict[str, int] = defaultdict(int)
        for record in selected:
            conversation_id = str(record.get("conversation_id", ""))
            if counts[conversation_id] >= per_dialogue_limit:
                continue
            limited.append(record)
            counts[conversation_id] += 1
            if limit is not None and len(limited) >= limit:
                break
        selected = limited
    elif limit is not None:
        selected = selected[:limit]
    return selected


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    """CLIエントリポイント。"""
    args = apply_bayes_model_defaults(parse_args())
    records = read_jsonl(args.input)
    selected = select_high_posterior_records(
        records,
        min_posterior=args.min_posterior,
        max_records=args.max_records,
        sort_by_posterior=args.sort_by_posterior,
        min_context_turns=args.min_context_turns,
        target_records=args.target_records,
        per_dialogue_limit=args.per_dialogue_limit,
        prefer_states=args.prefer_states,
        prefer_observations=args.prefer_observations,
        low_priority_states=args.low_priority_states,
        exclude_states=args.exclude_states,
        exclude_observations=args.exclude_observations,
        require_preferred=args.require_preferred,
        sort_by_selection=args.sort_by_selection,
    )
    if args.dry_run:
        print("高posterior抽出 dry-run")
        print(f"  input_records: {len(records)}")
        print(f"  selected_records: {len(selected)}")
        print(f"  min_posterior: {args.min_posterior}")
        print(f"  min_context_turns: {args.min_context_turns}")
        print(f"  target_records: {args.target_records}")
        print(f"  per_dialogue_limit: {args.per_dialogue_limit}")
        if args.bayes_model:
            print(f"  bayes_model: {args.bayes_model}")
            print(f"  prefer_states: {args.prefer_states}")
            print(f"  prefer_observations: {args.prefer_observations}")
            print(f"  exclude_states: {args.exclude_states}")
            print(f"  exclude_observations: {args.exclude_observations}")
        return 0
    write_jsonl(selected, args.output)
    print(f"高posterior応答を書き出しました: {args.output} ({len(selected)} 件)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
