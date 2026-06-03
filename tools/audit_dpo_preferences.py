"""DPO preferenceデータをgpt-5.4-proで品質監査し、学習用にフィルタする。"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tools.analyze_small_corpus import (
    OpenAIResponsesGenerator,
    TextGenerator,
    extract_json_object,
    load_env_file,
    resolve_analysis_model,
    write_json,
)
from tools.score_dialogue_with_transition_bayes_model import is_content_filter_error
from tools.translate_and_generate_dpo import bayes_model_version


DEFAULT_INPUT_PATH = "artifacts/datasets/dailydialog_ja_dpo_preferences.jsonl"
DEFAULT_OUTPUT_PATH = "artifacts/datasets/dailydialog_ja_dpo_preferences_audited.jsonl"
DEFAULT_REPORT_PATH = "artifacts/datasets/dailydialog_ja_dpo_preferences.audit.jsonl"
DEFAULT_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_MIN_QUALITY_SCORE = 0.78
DEFAULT_MAX_OUTPUT_TOKENS = 2048
PROMPT_TEMPLATE_VERSION = "audit_dpo_preferences.v1"


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="DPO JSONLを品質監査し、合格データだけを出力します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力DPO JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"合格DPO JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument("--audit-report", default=DEFAULT_REPORT_PATH, help=f"監査結果JSONL（既定: {DEFAULT_REPORT_PATH}）。")
    parser.add_argument("--bayes-model", default=DEFAULT_MODEL_PATH, help=f"ベイズモデルJSON（既定: {DEFAULT_MODEL_PATH}）。")
    parser.add_argument("--model", default=resolve_analysis_model(), help="品質監査モデル。品質優先ではgpt-5.4-proを推奨。")
    parser.add_argument("--min-quality-score", type=float, default=DEFAULT_MIN_QUALITY_SCORE, help="合格に必要な総合品質スコア。")
    parser.add_argument("--max-records", type=int, default=None, help="監査対象件数の上限。")
    parser.add_argument("--workers", type=int, default=1, help="サンプル単位で並列監査するworker数。")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、件数だけ確認します。")
    return parser.parse_args()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """JSONLを読み込む。"""
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{line_number}行目をJSONとして読めません: {exc}") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSONLが見つかりません: {input_path}") from exc
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path | str) -> None:
    """JSONLを書き出す。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: Path | str) -> None:
    """JSONLへ1レコード追記する。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dpo_key(record: dict[str, Any]) -> tuple[str, int]:
    """監査再開用のキーを返す。"""
    return str(record.get("source_dialogue_id", "")), int(record.get("turn_index", 0))


def build_audit_instructions() -> str:
    """DPO品質監査用の指示を作る。"""
    return (
        "あなたは研究用DPOデータの品質監査者です。"
        "目的は、回想支援型の会話能力をQwenに学習させるため、prompt/chosen/rejectedの品質を厳密に判定することです。\n\n"
        "合格条件:\n"
        "1. promptは自然な日本語の会話文脈として読める。\n"
        "2. chosenは文脈を受け止め、相手の過去の経験、思い出の情景、当時の気持ち、人間関係、行動、感覚的細部のいずれかを自然に深めている。\n"
        "3. chosenは直訳調、不自然な話者ラベル、説明文調、過度に長い応答になっていない。\n"
        "4. rejectedは同じpromptへの返答として一見自然だが、chosenよりも思い出の具体化・追想の深まり・会話戦略が明確に弱い。\n"
        "5. rejectedは文法破綻、攻撃性、安全性問題、意味不明さで低品質にしていない。\n"
        "6. chosenとrejectedの差がDPO学習に有効で、単なる言い換えや長さ違いではない。\n\n"
        "不合格にする例:\n"
        "- chosenが一般論、助言、情報説明、話題転換に寄っている。\n"
        "- chosenが回想を深めず、ただ共感しているだけである。\n"
        "- rejectedが不自然すぎて比較対象として弱い。\n"
        "- prompt/chosen/rejectedの会話文脈が噛み合っていない。\n"
        "- 翻訳により元の会話意図や会話戦略が崩れている。\n\n"
        "出力はJSONのみです。必須キーは pass, quality_score, chosen_alignment_score, rejected_contrast_score, japanese_naturalness_score, issues, reason です。"
        "passは真偽値、各scoreは0.0〜1.0、issuesは短い文字列配列、reasonは日本語で簡潔に書いてください。"
    )


def build_audit_input(record: dict[str, Any], *, index: int) -> str:
    """監査LLMへ渡す入力を作る。"""
    return (
        "json output only.\n"
        f"record_index: {index}\n"
        f"source_dialogue_id: {record.get('source_dialogue_id')}\n"
        f"turn_index: {record.get('turn_index')}\n"
        f"score_chosen: {record.get('score_chosen')}\n"
        f"score_rejected: {record.get('score_rejected')}\n"
        f"score_gap: {record.get('score_gap')}\n"
        f"state_sequence: {json.dumps(record.get('state_sequence', []), ensure_ascii=False)}\n"
        f"strategy_sequence: {json.dumps(record.get('strategy_sequence', []), ensure_ascii=False)}\n\n"
        f"prompt:\n{record.get('prompt', '')}\n\n"
        f"chosen:\n{record.get('chosen', '')}\n\n"
        f"rejected:\n{record.get('rejected', '')}"
    )


def _score(payload: dict[str, Any], key: str) -> float:
    """0.0〜1.0のスコアを取り出す。"""
    value = payload.get(key, 0.0)
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` は数値である必要があります。")
    return max(0.0, min(1.0, float(value)))


def parse_audit_payload(payload: dict[str, Any], *, min_quality_score: float) -> dict[str, Any]:
    """監査LLM出力を検証し、合否を確定する。"""
    raw_pass = payload.get("pass", False)
    if not isinstance(raw_pass, bool):
        raise ValueError("`pass` は真偽値である必要があります。")
    quality_score = _score(payload, "quality_score")
    chosen_alignment_score = _score(payload, "chosen_alignment_score")
    rejected_contrast_score = _score(payload, "rejected_contrast_score")
    japanese_naturalness_score = _score(payload, "japanese_naturalness_score")
    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        raise ValueError("`issues` は配列である必要があります。")
    reason = str(payload.get("reason", "")).strip()
    passed = (
        raw_pass
        and quality_score >= min_quality_score
        and chosen_alignment_score >= 0.70
        and rejected_contrast_score >= 0.65
        and japanese_naturalness_score >= 0.75
    )
    return {
        "pass": passed,
        "model_pass": raw_pass,
        "quality_score": quality_score,
        "chosen_alignment_score": chosen_alignment_score,
        "rejected_contrast_score": rejected_contrast_score,
        "japanese_naturalness_score": japanese_naturalness_score,
        "issues": [str(item) for item in issues if str(item).strip()],
        "reason": reason,
    }


def audit_one_record(
    record: dict[str, Any],
    *,
    index: int,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    min_quality_score: float,
) -> dict[str, Any]:
    """1件のDPOレコードを監査する。"""
    try:
        output_text = generator.generate(
            instructions=build_audit_instructions(),
            input_text=build_audit_input(record, index=index),
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
        audit = parse_audit_payload(extract_json_object(output_text), min_quality_score=min_quality_score)
    except Exception as exc:
        if not is_content_filter_error(exc):
            raise
        audit = {
            "pass": False,
            "model_pass": False,
            "quality_score": 0.0,
            "chosen_alignment_score": 0.0,
            "rejected_contrast_score": 0.0,
            "japanese_naturalness_score": 0.0,
            "issues": ["content_filter"],
            "reason": f"品質監査APIのcontent filterにより監査できませんでした: {exc}",
        }
    return {
        **record,
        "audit_quality_score": audit["quality_score"],
        "quality_audit": audit,
        "model_used_for_quality_audit": model,
        "quality_audit_prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }


def audit_records(
    records: list[dict[str, Any]],
    *,
    generator: TextGenerator,
    model: str,
    max_output_tokens: int,
    min_quality_score: float,
    max_records: int | None,
    workers: int,
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
    existing_accepted: list[dict[str, Any]] | None = None,
    existing_reports: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """DPOレコード群を監査し、合格レコードだけ返す。"""
    accepted = list(existing_accepted or [])
    audited_keys = {dpo_key(record) for record in accepted}
    audited_keys.update(dpo_key(record) for record in (existing_reports or []))
    source_records = records[:max_records] if max_records is not None else records
    pending = [record for record in source_records if dpo_key(record) not in audited_keys]
    print(
        f"[STEP 5/8] audit resume: accepted={len(accepted)} already_audited={len(audited_keys)} pending={len(pending)}",
        flush=True,
    )

    def handle_result(audited: dict[str, Any]) -> None:
        if report_path is not None:
            append_jsonl(
                {
                    "source_dialogue_id": audited.get("source_dialogue_id"),
                    "turn_index": audited.get("turn_index"),
                    "pass": audited["quality_audit"]["pass"],
                    "audit_quality_score": audited["audit_quality_score"],
                    "quality_audit": audited["quality_audit"],
                    "model_used_for_quality_audit": model,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                },
                report_path,
            )
        if audited["quality_audit"]["pass"]:
            accepted.append(audited)
            if output_path is not None:
                append_jsonl(audited, output_path)
            print(
                f"[STEP 5/8] audit accepted quality={audited['audit_quality_score']:.3f} "
                f"{audited.get('source_dialogue_id')}#{audited.get('turn_index')}",
                flush=True,
            )
        else:
            print(
                f"[STEP 5/8] audit rejected quality={audited['audit_quality_score']:.3f} "
                f"{audited.get('source_dialogue_id')}#{audited.get('turn_index')}",
                flush=True,
            )

    indexed = list(enumerate(pending, start=1))
    if workers <= 1:
        for index, record in indexed:
            print(f"[STEP 5/8] audit: {index}/{len(indexed)} {dpo_key(record)}", flush=True)
            handle_result(
                audit_one_record(
                    record,
                    index=index,
                    generator=generator,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    min_quality_score=min_quality_score,
                )
            )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    audit_one_record,
                    record,
                    index=index,
                    generator=generator,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    min_quality_score=min_quality_score,
                ): record
                for index, record in indexed
            }
            for future in as_completed(futures):
                source = futures[future]
                audited = future.result()
                completed += 1
                progress = completed / len(indexed) * 100.0 if indexed else 100.0
                print(
                    f"[STEP 5/8] audit: {completed}/{len(indexed)} ({progress:.1f}%) {dpo_key(source)}",
                    flush=True,
                )
                handle_result(audited)
    return sorted(accepted, key=lambda record: float(record.get("score_gap", 0.0)), reverse=True)


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    records = read_jsonl(args.input)
    if args.dry_run:
        print("DPO品質監査 dry-run")
        print(f"  input_records: {len(records)}")
        print(f"  model: {args.model}")
        print(f"  min_quality_score: {args.min_quality_score}")
        print(f"  max_records: {args.max_records}")
        return 0

    accepted = audit_records(
        records,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        min_quality_score=args.min_quality_score,
        max_records=args.max_records,
        workers=max(1, args.workers),
        output_path=args.output,
        report_path=args.audit_report,
        existing_accepted=read_jsonl(args.output) if Path(args.output).exists() else [],
        existing_reports=read_jsonl(args.audit_report) if Path(args.audit_report).exists() else [],
    )
    write_jsonl(accepted, args.output)
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    write_json(
        {
            "input": args.input,
            "output": args.output,
            "audit_report": args.audit_report,
            "audit_model": args.model,
            "bayes_model": args.bayes_model,
            "bayes_model_version": bayes_model_version(args.bayes_model) if Path(args.bayes_model).exists() else "",
            "min_quality_score": args.min_quality_score,
            "max_records": args.max_records,
            "workers": max(1, args.workers),
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "records_written": len(accepted),
        },
        manifest_path,
    )
    print(f"監査済みDPO JSONLを書き出しました: {args.output} ({len(accepted)} 件)")
    print(f"監査manifestを書き出しました: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
