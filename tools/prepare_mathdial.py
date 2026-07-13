"""MathDialをBASiS共通形式へ変換するCLI。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.dialogue_schema import canonical_json_hash
from tools.mathdial_dataset import (
    DEFAULT_CONFIG_PATH,
    download_mathdial_files,
    load_yaml_config,
    prepare_mathdial,
    read_jsonl,
    sha256_file,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/mathdial_wildchat")


def parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="MathDialをBASiS共通会話形式へ変換します。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-name", default="", help="設定のdataset_nameを上書きします。")
    parser.add_argument("--revision", default="", help="設定のrevisionを上書きします。")
    parser.add_argument("--validation-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="成果物を書かず統計だけ表示します。")
    return parser.parse_args()


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """JSONLを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    """整形JSONを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_report_markdown(summary: dict[str, Any]) -> str:
    """前処理統計のMarkdownを作る。"""
    raw = summary["raw"]
    normalized = summary["normalized_all"]
    operations = summary["normalization"]
    samples = summary["samples"]
    duplicate_audit = summary["exact_duplicate_audit"]
    source_split_operations = summary["normalization_by_source_split"]
    lines = [
        "# MathDial前処理レポート",
        "",
        "## 全体",
        "",
        "| 指標 | 正規化前 | 正規化後 |",
        "|---|---:|---:|",
        f"| 会話数 | {raw['total_conversations']} | {normalized['conversations']} |",
        "| 平均発話数 | "
        f"{raw['average_messages_per_conversation']:.3f} | "
        f"{normalized['average_messages_per_conversation']:.3f} |",
        f"| user発話数 | {raw['user_turns']} | {normalized['user_turns']} |",
        f"| assistant発話数 | {raw['assistant_turns']} | {normalized['assistant_turns']} |",
        f"| user平均文字数 | {raw['average_user_characters']:.3f} | {normalized['average_user_characters']:.3f} |",
        "| assistant平均文字数 | "
        f"{raw['average_assistant_characters']:.3f} | "
        f"{normalized['average_assistant_characters']:.3f} |",
        "",
        "## 正規化操作",
        "",
        f"- 空発話除去: {operations['empty_segments_removed']}",
        f"- 連続assistant境界: {operations['consecutive_assistant_boundaries']}",
        f"- 連結対象assistant group: {operations['assistant_merge_groups']}",
        f"- 完全一致assistant重複除去: {operations['identical_assistant_segments_removed']}",
        f"- 未知の括弧prefix保持: {operations['unknown_teacher_move_prefixes']}",
        "",
        "### 連続発話カウントの対応",
        "",
        "- 空発話除去前の連続Teacher境界: "
        f"{operations['consecutive_assistant_boundaries_before_empty_removal']} "
        f"(train={source_split_operations['train']['consecutive_assistant_boundaries_before_empty_removal']}, "
        f"test={source_split_operations['test']['consecutive_assistant_boundaries_before_empty_removal']})",
        "- 空発話除去後の連続assistant境界: "
        f"{operations['consecutive_assistant_boundaries']}",
        "- assistant発話数の減少: "
        f"{operations['consecutive_assistant_boundaries']} "
        "(各groupのsource segment数-1の合計)",
        "- 連結group数: "
        f"{duplicate_audit['assistant_merge_groups']} / group size分布: "
        f"{duplicate_audit['assistant_merge_group_size_distribution']}",
        "- assistant→空assistant→user: "
        f"{operations['empty_assistant_between_assistant_and_user']}",
        "- assistant→空assistant→assistant: "
        f"{operations['empty_assistant_between_assistants']}",
        "- user→空assistant→user: "
        f"{operations['empty_assistant_between_users']}",
        "",
        "## 完全一致重複監査",
        "",
        f"- 完全一致境界: {duplicate_audit['exact_duplicate_boundaries']}",
        f"- 一意な重複本文: {duplicate_audit['unique_duplicate_texts']}",
        f"- teacher move一致 / 不一致: {duplicate_audit['same_teacher_move']} / "
        f"{duplicate_audit['different_teacher_move']}",
        "- 文字数 min / median / mean / max: "
        f"{duplicate_audit['minimum_characters']} / {duplicate_audit['median_characters']} / "
        f"{duplicate_audit['mean_characters']} / {duplicate_audit['maximum_characters']}",
        "- 完全一致判定はStudent発話を挟まない隣接Teacher segmentだけを対象とする。",
        "- 片方の本文だけを残すが、両方の元位置とteacher moveはmetadataへ保持する。",
        "",
        "## リーク防止とサンプル",
        "",
        f"- train/test重複qid: {raw['overlap_qids']}",
        f"- quarantine会話: {raw['train_conversations_quarantined']}",
        f"- assistantサンプル: {samples['total']}",
        f"- after state観測可能: {samples['after_state_observed']}",
        f"- after state未観測: {samples['after_state_unobserved']}",
        f"- DPO適格: {samples['dpo_eligible']}",
        "",
        "## Split",
        "",
        "| split | conversations | messages | assistant samples |",
        "|---|---:|---:|---:|",
    ]
    for split, split_summary in summary["by_split"].items():
        lines.append(
            f"| {split} | {split_summary['conversations']} | "
            f"{split_summary['messages']} | {split_summary['assistant_turns']} |"
        )
    lines.extend(["", "## 完全一致重複の例", ""])
    for index, example in enumerate(duplicate_audit["examples"], start=1):
        lines.extend(
            [
                f"### Example {index}",
                "",
                f"- source: {example['source_split']} / qid={example['qid']} / "
                f"turns={example['source_turn_indices']}",
                f"- teacher moves: {example['teacher_moves']}",
                f"- previous user: {example['previous_user']}",
                f"- duplicated assistant: {example['duplicate_assistant_text']}",
                f"- next user: {example['next_user']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _print_summary(summary: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n少数サンプル:")
    for sample in samples[:3]:
        print(
            json.dumps(
                {
                    "sample_id": sample["sample_id"],
                    "history": sample["history"][-2:],
                    "response": sample["response"][:240],
                    "next_user_turn": (
                        sample["next_user_turn"][:240] if sample["next_user_turn"] else None
                    ),
                    "metadata": sample["metadata"],
                },
                ensure_ascii=False,
            )
        )


def main() -> int:
    """MathDial前処理を実行する。"""
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.dataset_name:
        config["dataset_name"] = args.dataset_name
    if args.revision:
        config["revision"] = args.revision
    if args.validation_ratio is not None:
        config["validation_ratio"] = args.validation_ratio
    if args.seed is not None:
        config["seed"] = args.seed

    source_paths, source_metadata = download_mathdial_files(
        str(config["dataset_name"]),
        str(config["revision"]),
    )
    rows_by_split = {split: read_jsonl(path) for split, path in source_paths.items()}
    prepared = prepare_mathdial(rows_by_split, config=config)
    _print_summary(prepared.summary, prepared.samples)
    if args.dry_run:
        return 0

    data_dir = args.output_root / "data"
    report_dir = args.output_root / "reports"
    conversations_path = data_dir / "mathdial_conversations.jsonl"
    samples_path = data_dir / "mathdial_assistant_samples.jsonl"
    quarantine_path = data_dir / "mathdial_qid_overlap_quarantine.jsonl"
    summary_path = report_dir / "preprocessing_summary.json"
    report_path = report_dir / "preprocessing_report.md"
    manifest_path = data_dir / "manifest.json"
    write_jsonl(prepared.conversations, conversations_path)
    write_jsonl(prepared.samples, samples_path)
    write_jsonl(prepared.quarantine, quarantine_path)
    write_json(prepared.summary, summary_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report_markdown(prepared.summary), encoding="utf-8")

    manifest = {
        "pipeline": "mathdial_preprocessing_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_metadata,
        "license_notice": {
            "huggingface_card": source_metadata.get("huggingface_license"),
            "official_github_readme": "CC BY-SA 4.0",
            "note": "公式配布元間で表記が異なるため、再配布前に条件を再確認する。",
        },
        "config_path": str(args.config),
        "config": config,
        "config_sha256": canonical_json_hash(config),
        "outputs": {},
        "summary": prepared.summary,
    }
    for key, path in {
        "conversations": conversations_path,
        "assistant_samples": samples_path,
        "quarantine": quarantine_path,
        "summary": summary_path,
        "report": report_path,
    }.items():
        manifest["outputs"][key] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    write_json(manifest, manifest_path)
    print(f"MathDial前処理成果物を書き出しました: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
