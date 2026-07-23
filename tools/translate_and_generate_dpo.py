"""抽出済み対話を自然な日本語DPOデータへ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.dpo_prompting import (
    DPO_PROMPT_TEMPLATE_VERSION,
    MEDITOD_DPO_PROMPT_TEMPLATE_VERSION,
    build_dpo_prompt_from_context_text,
    build_mathdial_dpo_prompt_from_context_text,
    build_meditod_dpo_prompt_from_context_text,
)
from core.transition_bayes_model import (
    TransitionBayesModel,
    load_transition_bayes_model,
)
from tools.analyze_small_corpus import (
    TextGenerator,
    extract_json_object,
    resolve_analysis_model,
    write_json,
)
from tools.audit_logging import DEFAULT_AUDIT_LOG_PATH, append_audit_log
from tools.score_dialogue_with_bayes_model import (
    OpenAIResponsesGenerator,
    load_env_file,
    resolve_scoring_model,
)
from tools.score_dialogue_with_transition_bayes_model import (
    build_transition_scoring_instructions,
    is_content_filter_error,
    sanitize_text_for_content_filter_retry,
    score_single_record,
)
from tools.jsonl_utils import ensure_jsonl_append_boundary, read_jsonl_records


DEFAULT_INPUT_PATH = "artifacts/datasets/dailydialog_selected_en.jsonl"
DEFAULT_BAYES_MODEL_PATH = "artifacts/bayes_models/generated_transition_bayes_model.json"
DEFAULT_OUTPUT_PATH = "artifacts/datasets/dailydialog_ja_dpo_preferences.jsonl"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_CANDIDATES = 4
DEFAULT_MIN_SCORE_GAP = 0.25
DEFAULT_MIN_CHOSEN_POSTERIOR = 0.70
DEFAULT_MAX_REJECTED_POSTERIOR = 0.55
DEFAULT_GAP_RESCUE_MAX_REJECTED_POSTERIOR: float | None = None
DEFAULT_GAP_RESCUE_MIN_SCORE_GAP: float | None = None
DEFAULT_SEED = 42
DEFAULT_STYLE_PRESET = "reminiscence"
PROMPT_TEMPLATE_VERSION = "translate_and_generate_dpo.v2"
MATHDIAL_NUMERIC_FIDELITY_VERSION = "mathdial_numeric_fidelity.v2"
MEDITOD_MEDICAL_FIDELITY_VERSION = "meditod_medical_fidelity.v2"
NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?%?"
)
SKIP_ADMIN_KEYS = {
    "skip_reason",
    "skipped_at",
    "acceptance_thresholds",
    "would_accept_with_current_thresholds",
}


@dataclass
class DpoGenerationStats:
    """DPO生成のskip理由をmanifestへ残すための集計。"""

    existing_records: int = 0
    attempted: int = 0
    accepted_new: int = 0
    skipped_low_chosen: int = 0
    skipped_high_rejected: int = 0
    skipped_small_gap: int = 0
    skipped_content_filter_generation: int = 0
    skipped_invalid_generation: int = 0
    skipped_sample_error: int = 0
    skipped_source_too_long: int = 0

    def as_dict(self) -> dict[str, int]:
        """JSONへ書けるdictに変換する。"""
        return {
            "existing_records": self.existing_records,
            "attempted": self.attempted,
            "accepted_new": self.accepted_new,
            "skipped_low_chosen": self.skipped_low_chosen,
            "skipped_high_rejected": self.skipped_high_rejected,
            "skipped_small_gap": self.skipped_small_gap,
            "skipped_content_filter_generation": self.skipped_content_filter_generation,
            "skipped_invalid_generation": self.skipped_invalid_generation,
            "skipped_sample_error": self.skipped_sample_error,
            "skipped_source_too_long": self.skipped_source_too_long,
        }


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    load_env_file()
    parser = argparse.ArgumentParser(description="抽出済み英語応答から日本語DPO JSONLを作成します。")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help=f"入力JSONL（既定: {DEFAULT_INPUT_PATH}）。")
    parser.add_argument("--bayes-model", default=DEFAULT_BAYES_MODEL_PATH, help=f"状態遷移ベイズモデルJSON（既定: {DEFAULT_BAYES_MODEL_PATH}）。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"出力DPO JSONL（既定: {DEFAULT_OUTPUT_PATH}）。")
    parser.add_argument(
        "--skipped-output",
        default=None,
        help="条件未満でskipした候補の詳細JSONL。未指定なら出力DPOと同じ場所に *_skipped.jsonl を作ります。",
    )
    parser.add_argument("--model", default=resolve_scoring_model(), help="翻訳・rejected生成モデル。大量処理ではgpt-5.4を推奨。")
    parser.add_argument("--audit-model", default=resolve_analysis_model(), help="必要時の品質監査モデル。既定はgpt-5.4-pro系。")
    parser.add_argument("--score-model", default=resolve_scoring_model(), help="再スコアリングモデル。")
    parser.add_argument(
        "--style-preset",
        choices=("reminiscence", "esconv_support", "mathdial_tutoring", "meditod_history_taking"),
        default=DEFAULT_STYLE_PRESET,
        help="翻訳・rejected生成の方針。ESConvではesconv_supportを指定します。",
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大出力トークン数。")
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES, help="rejected候補の生成数。")
    parser.add_argument("--min-score-gap", type=float, default=DEFAULT_MIN_SCORE_GAP, help="採用するscore_gapの下限。")
    parser.add_argument("--min-chosen-posterior", type=float, default=DEFAULT_MIN_CHOSEN_POSTERIOR, help="翻訳後chosenのposterior下限。")
    parser.add_argument("--max-rejected-posterior", type=float, default=DEFAULT_MAX_REJECTED_POSTERIOR, help="rejectedのposterior上限。")
    parser.add_argument(
        "--gap-rescue-max-rejected-posterior",
        type=float,
        default=DEFAULT_GAP_RESCUE_MAX_REJECTED_POSTERIOR,
        help="score_gapが十分大きい場合に許容するrejected posterior上限。未指定なら救済条件を使いません。",
    )
    parser.add_argument(
        "--gap-rescue-min-score-gap",
        type=float,
        default=DEFAULT_GAP_RESCUE_MIN_SCORE_GAP,
        help="rejected上限を緩める場合に必要なscore_gap下限。未指定なら救済条件を使いません。",
    )
    parser.add_argument("--max-records", type=int, default=None, help="処理件数の上限。")
    parser.add_argument(
        "--max-source-characters",
        type=int,
        default=None,
        help="完全なprompt+responseの最大文字数。超過サンプルはAPI呼び出し前に除外します。",
    )
    parser.add_argument("--target-records", type=int, default=None, help="accepted DPO件数の目標。達したら処理を終了します。")
    parser.add_argument(
        "--allow-target-shortfall",
        action="store_true",
        help="入力を処理し切って目標未達でも部分成果物を保存して正常終了します。候補追加型pipeline専用です。",
    )
    parser.add_argument("--workers", type=int, default=1, help="サンプル単位で並列生成するworker数。1なら逐次処理。")
    parser.add_argument(
        "--skip-sample-errors",
        action="store_true",
        help="個別サンプルのAPI/JSON/再スコア失敗をskipしてDPO生成を継続します。",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="再現性記録用の乱数シード。")
    parser.add_argument("--audit-log", default=DEFAULT_AUDIT_LOG_PATH, help="重要操作の要約を追記するaudit_log.mdのパス。")
    parser.add_argument("--heartbeat-file", default=None, help="長時間DPO生成の進捗を書き出すheartbeat JSON。")
    parser.add_argument("--heartbeat-stage-prefix", default="dpo_generation", help="heartbeat stage名の接頭辞。")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力件数だけ確認します。")
    return parser.parse_args()


def write_heartbeat(path: Path | str | None, stage: str, payload: dict[str, Any] | None = None) -> None:
    """長時間処理の生存確認用heartbeatを書き出す。"""
    if path is None:
        return
    heartbeat_path = Path(path)
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": stage,
    }
    if payload:
        body.update(payload)
    heartbeat_path.write_text(json.dumps(body, ensure_ascii=False) + "\n", encoding="utf-8")


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
        raise FileNotFoundError(f"入力JSONLが見つかりません: {input_path}") from exc
    if not records:
        raise ValueError("入力JSONLに有効なレコードがありません。")
    return records


def read_existing_dpo_records(path: Path | str) -> list[dict[str, Any]]:
    """既存DPO出力があれば読み込む。なければ空配列を返す。"""
    records, skipped = read_jsonl_records(
        path,
        missing_ok=True,
        strict=False,
        label="既存DPO出力",
    )
    if skipped:
        print(f"[WARN] 既存DPO出力の壊れた行をskipしました: skipped={skipped}", flush=True)
    best_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if "source_dialogue_id" not in record or "turn_index" not in record:
            print("[WARN] 既存DPO出力の再開キー欠落行をskipしました", flush=True)
            continue
        key = dpo_record_key(record)
        current = best_by_key.get(key)
        if current is not None:
            duplicate_count += 1
        if current is None or float(record.get("score_gap", 0.0)) > float(
            current.get("score_gap", 0.0)
        ):
            best_by_key[key] = record
    if duplicate_count:
        print(
            f"[WARN] 既存DPO出力の重複source keyをdedupeしました: duplicates={duplicate_count}",
            flush=True,
        )
    return list(best_by_key.values())


def default_skipped_output_path(output_path: Path | str) -> Path:
    """DPO出力パスからskip詳細JSONLの既定パスを作る。"""
    path = Path(output_path)
    return path.with_name(f"{path.stem}_skipped.jsonl")


def read_existing_skipped_records(path: Path | str | None) -> list[dict[str, Any]]:
    """既存skip詳細があれば読み込む。なければ空配列を返す。"""
    if path is None:
        return []
    records, skipped = read_jsonl_records(
        path,
        missing_ok=True,
        strict=False,
        label="既存skip詳細",
    )
    if skipped:
        print(f"[WARN] 既存skip詳細の壊れた行をskipしました: skipped={skipped}", flush=True)
    best_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if "source_dialogue_id" not in record or "turn_index" not in record:
            continue
        key = dpo_record_key(record)
        current = best_by_key.get(key)
        if current is not None:
            duplicate_count += 1
        if current is None or float(record.get("score_gap", -1.0)) > float(
            current.get("score_gap", -1.0)
        ):
            best_by_key[key] = record
    if duplicate_count:
        print(
            f"[WARN] 既存skip詳細の重複source keyをdedupeしました: duplicates={duplicate_count}",
            flush=True,
        )
    return list(best_by_key.values())


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
    ensure_jsonl_append_boundary(output_path)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def dpo_record_key(record: dict[str, Any]) -> tuple[str, int]:
    """DPO再開判定用のキーを返す。"""
    return str(record["source_dialogue_id"]), int(record["turn_index"])


def source_record_key(record: dict[str, Any]) -> tuple[str, int]:
    """抽出元レコードのキーを返す。"""
    return str(record["conversation_id"]), int(record["turn_index"])


def dpo_record_without_skip_admin(record: dict[str, Any]) -> dict[str, Any]:
    """skip管理用フィールドを除いたDPOレコードを返す。"""
    return {key: value for key, value in record.items() if key not in SKIP_ADMIN_KEYS}


def bayes_model_version(path: Path | str) -> str:
    """ベイズモデルJSONの内容ハッシュを返す。"""
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _style_specific_translation_policy(style_preset: str) -> str:
    """style presetごとの翻訳・rejected生成方針を返す。"""
    if style_preset == "mathdial_tutoring":
        return (
            "翻訳方針:\n"
            "- promptとchosenを、日本人の学習者と個別指導者の自然な日本語対話へ翻訳してください。\n"
            "- 数値、数式、単位、問題条件、学習者の誤りを保持し、誤答を翻訳時に訂正しないでください。\n"
            "- chosenに存在しない説明、ヒント、質問、最終解答を追加しないでください。\n"
            "- chosenが持つ診断、問い返し、焦点化、段階的ヒント、説明、確認の機能を忠実に保持してください。\n"
            "- 不自然な直訳は避けますが、BASiSらしさを翻訳によって強めないでください。\n"
            "- prompt内のUser/AIの順序と各発話の対応を変えないでください。\n\n"
            "rejected候補の生成方針:\n"
            "- rejectedは必ず同じtranslated_promptに対する日本語応答にしてください。\n"
            "- 文法的には自然で安全だが、個別指導としてchosenより弱い応答を作ってください。\n"
            "- 弱点は、誤りを診断しない、一般論だけを言う、焦点がずれる、足場を飛ばす、答えを早く教える、自己修正を促さない、の間で分散してください。\n"
            "- 問題や学習者発話にない新しい事実を捏造しないでください。\n"
            "- chosenの単なる言い換え、壊れた文章、攻撃的な文章は禁止です。"
        )
    if style_preset == "meditod_history_taking":
        return (
            "翻訳方針:\n"
            "- promptとchosenを、日本人の患者と医療者による自然な医療相談へ忠実に翻訳してください。\n"
            "- 否定、発症時期、期間、経過、重症度、数値、単位、薬剤名、症状名を省略・反転・補完しないでください。\n"
            "- 医学的に誤って見えても、原文にない診断、助言、red flag、安心づけを翻訳時に追加しないでください。\n"
            "- prompt内のUser/AIの順序と、chosenが持つ質問・要約・段階移行の機能を保持してください。\n"
            "- 薬剤名や固有の医学用語は、必要なら原語を括弧内に残してください。\n"
            "- BASiSらしさを翻訳によって強めないでください。\n\n"
            "rejected候補の生成方針:\n"
            "- rejectedは必ず同じtranslated_promptに対する日本語応答にしてください。\n"
            "- 攻撃的、危険、虚偽の診断や投薬指示は作らず、通常の会話として安全に読める応答にしてください。\n"
            "- 病歴聴取としてchosenより弱くなるよう、不足情報を聞かない、既に得た情報を重複して聞く、"
            "質問順が不自然、曖昧な一般論、情報不足で早くまとめる、の弱点を候補間で分散してください。\n"
            "- chosenの単なる言い換えや壊れた文章は禁止です。"
        )
    if style_preset == "esconv_support":
        return (
            "翻訳方針:\n"
            "- 直訳ではなく、日本人同士の自然な支援的対話として書き換えてください。\n"
            "- 元のchosenが持つESConv由来の支援機能を保持してください。特に、感情の受け止め・感情反映、状況の言い換え、確認質問、肯定・安心づけ、必要最小限の提案、情報提供のどれをしているかを見極めてください。\n"
            "- 相手の悩み・状況・感情語を具体的に拾い、最初に軽い共感または言い換えを入れてください。\n"
            "- 質問する場合は、相手が次を話しやすい確認質問を1つだけ入れてください。詰問調や原因追及に見える聞き方は禁止です。\n"
            "- 提案や情報提供を入れる場合は、感情を受け止めた後に、現実的で小さい一歩として短く添えてください。\n"
            "- 早すぎる断定、説教、一般論、長すぎる助言、相談の早すぎる終了、話題転換は避けてください。\n"
            "- promptは過去の会話文脈として自然に読めるよう日本語化し、話者ラベルはUser/AIに揃えるか、意味が分かる自然な形にしてください。\n"
            "- chosenは、文脈を受けた理想的な支援応答として自然で、短すぎず、説明的すぎない1〜2文にしてください。\n\n"
            "rejected候補の生成方針:\n"
            "- rejectedは同じtranslated_promptに対する返答として作ってください。\n"
            "- 文法的に破綻した返答、攻撃的な返答、安全性に問題がある返答は作らないでください。\n"
            "- 一見自然だが、推定された支援的対話目的・会話状態・観測ラベルに照らすと低評価になりやすい返答にしてください。\n"
            "- chosenの単なる短縮、同義表現、語尾だけの変更は禁止です。\n"
            "- 候補ごとに低評価になりやすい理由が少しずつ異なるようにしてください。例: 感情を反映しない、状況を言い換えない、一般論だけで返す、早すぎる助言に飛ぶ、相談者の具体的状況を拾わない、確認せず断定する、会話を早く閉じる、相手の不安を軽く扱う。\n"
            "- rejected_candidates_countが多い場合も、各候補の弱点を必ず分散してください。少なくとも、感情反映不足、状況の拾い漏れ、一般論への逃げ、早すぎる助言、確認なしの断定、会話の早期終了のいずれかが候補群に含まれるようにしてください。\n"
            "- rejectedは低品質な文章ではなく、普通の会話としては読めるがESConv支援応答としては明確に弱い返答にしてください。相談者の感情や具体的状況を少し外すことで、chosenとの差が出るようにしてください。\n"
            "- rejectedは、ベースモデルが出しがちな普通の返答に見えるが、ESConv支援Strategyの観点ではchosenより明確に弱いものにしてください。\n"
            "- rejectedも日本語としては自然で、DPO学習で比較対象にできる品質にしてください。"
        )
    return (
        "翻訳方針:\n"
        "- 直訳ではなく、日本人同士の自然な会話として書き換えてください。\n"
        "- ただし、元のchosenが持つ意図、感情の受け止め、話題の深め方、会話状態の進み方は保ってください。\n"
        "- 特に、相手の過去の経験、思い出の情景、当時の気持ち、一緒にいた人、季節・場所・音・匂いなどの具体性を深める応答戦略を保持してください。\n"
        "- chosenが質問の場合は、尋問調ではなく、相手が思い出を話しやすい一問にしてください。\n"
        "- promptは過去の会話文脈として自然に読めるよう、話者ラベルも含めて日本語化してください。\n"
        "- chosenは、文脈を受けた理想的な次の応答として自然で、短すぎず、説明的すぎず、会話を続けやすい表現にしてください。\n\n"
        "rejected候補の生成方針:\n"
        "- rejectedは同じtranslated_promptに対する返答として作ってください。\n"
        "- 文法的に破綻した返答、攻撃的な返答、安全性に問題がある返答は作らないでください。\n"
        "- 一見自然に読めるが、推定されたデータセット目的・会話状態・観測ラベルに照らすと低評価になりやすい返答にしてください。\n"
        "- chosenの単なる短縮、同義表現、語尾だけの変更は禁止です。\n"
        "- 候補ごとに低評価になりやすい理由が少しずつ異なるようにしてください。例: 文脈を浅く流す、一般論に戻す、助言に逸れる、相手の具体的内容を拾わない、昔の経験や情景を深めない、会話を早く閉じる。\n"
        "- rejectedも日本語としては自然で、DPO学習で比較対象にできる品質にしてください。"
    )


def build_translation_rejected_instructions(
    model: TransitionBayesModel,
    *,
    style_preset: str = DEFAULT_STYLE_PRESET,
) -> str:
    """翻訳とrejected候補生成の指示を作る。"""
    state_lines = "\n".join(f"- {name}: {model.state_descriptions[name]}" for name in model.states)
    observation_lines = "\n".join(f"- {name}: {model.observation_descriptions[name]}" for name in model.observations)
    style_policy = _style_specific_translation_policy(style_preset)
    return (
        "あなたはDPO学習用の日本語対話データ作成者です。"
        "目的は、英語の文脈付き高スコア応答を、日本人同士の自然な会話として使えるDPOサンプルへ変換することです。"
        "このデータはローカルLLMに小コーパス由来の会話戦略を学習させるために使われます。\n\n"
        f"style_preset: {style_preset}\n\n"
        f"{style_policy}\n\n"
        "出力はJSONのみで、translated_prompt, translated_chosen, rejected_candidates, translation_quality_score を含めてください。"
        "rejected_candidatesは文字列配列です。translation_quality_scoreは、意図・感情・会話戦略を保持できている度合いを0.0〜1.0で付けてください。\n\n"
        f"推定されたデータセット目的:\n{model.dataset_hypothesis}\n\n"
        f"会話状態:\n{state_lines}\n\n"
        f"観測ラベル:\n{observation_lines}"
    )


def build_translation_rejected_input(record: dict[str, Any], *, candidates: int, seed: int) -> str:
    """翻訳・rejected生成用の入力を作る。"""
    return (
        f"json output only.\n"
        f"seed: {seed}\n"
        f"rejected_candidates_count: {candidates}\n"
        f"source_dialogue_id: {record.get('conversation_id')}\n"
        f"turn_index: {record.get('turn_index')}\n\n"
        f"english_prompt:\n{record.get('prompt', '')}\n\n"
        f"english_chosen_response:\n{record.get('response', '')}"
    )


def build_safe_translation_rejected_input(
    record: dict[str, Any],
    *,
    candidates: int,
    seed: int,
) -> str:
    """content_filter再試行用に具体情報を中立化した入力を作る。"""
    safe_record = {
        **record,
        "prompt": sanitize_text_for_content_filter_retry(str(record.get("prompt", ""))),
        "response": sanitize_text_for_content_filter_retry(str(record.get("response", ""))),
    }
    return (
        "content filterの誤検出を避けるため、固有の年齢・日付・個人名・親密表現などを"
        "中立的なプレースホルダに置換した安全化版です。"
        "評価では、置換された具体情報そのものではなく、会話文脈への応答戦略を保ってください。\n\n"
        + build_translation_rejected_input(safe_record, candidates=candidates, seed=seed)
    )


def validate_translation_payload(payload: dict[str, Any], *, candidates: int) -> dict[str, Any]:
    """翻訳・rejected生成のJSONを検証する。"""
    translated_prompt = str(payload.get("translated_prompt", "")).strip()
    translated_chosen = str(payload.get("translated_chosen", "")).strip()
    rejected_candidates = payload.get("rejected_candidates")
    if not translated_prompt:
        raise ValueError("`translated_prompt` が空です。")
    if not translated_chosen:
        raise ValueError("`translated_chosen` が空です。")
    if not isinstance(rejected_candidates, list):
        raise ValueError("`rejected_candidates` は配列である必要があります。")
    rejected_texts = []
    seen = set()
    for item in rejected_candidates:
        text = str(item).strip()
        if not text or text == translated_chosen or text in seen:
            continue
        rejected_texts.append(text)
        seen.add(text)
    if len(rejected_texts) < candidates:
        raise ValueError("rejected候補数が不足しています。")
    quality = payload.get("translation_quality_score", 0.0)
    if not isinstance(quality, (int, float)):
        raise ValueError("`translation_quality_score` は数値である必要があります。")
    return {
        "translated_prompt": translated_prompt,
        "translated_chosen": translated_chosen,
        "rejected_candidates": rejected_texts,
        "translation_quality_score": max(0.0, min(1.0, float(quality))),
    }


def normalize_numeric_token(token: str) -> str:
    """桁区切りの差を吸収し、数値・分数tokenを比較可能にする。"""
    suffix = "%" if token.endswith("%") else ""
    body = token[:-1] if suffix else token
    normalized_parts: list[str] = []
    for part in body.split("/"):
        compact = part.replace(",", "")
        try:
            value = Decimal(compact)
            normalized = format(value.normalize(), "f")
            normalized_parts.append("0" if normalized in ("-0", "+0") else normalized)
        except InvalidOperation:
            normalized_parts.append(compact)
    return "/".join(normalized_parts) + suffix


def extract_normalized_numeric_tokens(text: str) -> list[str]:
    """出現順を保ったまま正規化済み数値・分数tokenを抽出する。"""
    canonical = text.replace("％", "%")
    canonical = re.sub(r"(\d[\d,]*)\s*分の\s*(\d[\d,]*)", r"\2/\1", canonical)
    canonical = re.sub(r"(\d{1,2})\s*時\s*(\d{1,2})\s*分", r"\1.\2", canonical)
    return [normalize_numeric_token(token) for token in NUMERIC_TOKEN_PATTERN.findall(canonical)]


def missing_mathdial_numeric_tokens(source_text: str, translated_text: str) -> list[str]:
    """翻訳側に意味的に存在しない一意な数値・数式tokenを返す。"""
    source_tokens = list(dict.fromkeys(extract_normalized_numeric_tokens(source_text)))
    translated_tokens = set(extract_normalized_numeric_tokens(translated_text))
    translated_components = {
        component.rstrip("%")
        for token in translated_tokens
        for component in token.rstrip("%").split("/")
    }
    missing: list[str] = []
    for token in source_tokens:
        if token in translated_tokens:
            continue
        # 15/100のような式が保たれていれば、周辺説明の単独15や100は欠落としない。
        if "/" not in token and token.rstrip("%") in translated_components:
            continue
        missing.append(token)
    return missing


MEDICAL_CITATION_PATTERN = re.compile(
    r"\[\s*\d+(?:\s*[-–—]\s*\d+)?"
    r"(?:\s*[,;]\s*\d+(?:\s*[-–—]\s*\d+)?)*\s*\]"
)


def remove_medical_citation_numbers(text: str) -> str:
    """論文の参照番号を除き、患者情報として意味を持つ数値だけを検査する。"""
    return MEDICAL_CITATION_PATTERN.sub("", text)


def missing_meditod_numeric_tokens(
    source_text: str,
    translated_text: str,
) -> list[str]:
    """引用番号を除外してMediTOD翻訳の数値保持を検査する。"""
    return missing_mathdial_numeric_tokens(
        remove_medical_citation_numbers(source_text),
        remove_medical_citation_numbers(translated_text),
    )


def mathdial_translation_fidelity_errors(
    source_record: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, list[str]]:
    """promptとchosenを分けて数値保持違反を検出する。"""
    errors = {
        "prompt": missing_mathdial_numeric_tokens(
            str(source_record.get("prompt", "")),
            str(payload["translated_prompt"]),
        ),
        "chosen": missing_mathdial_numeric_tokens(
            str(source_record.get("response", "")),
            str(payload["translated_chosen"]),
        ),
    }
    return {field: tokens for field, tokens in errors.items() if tokens}


def validate_mathdial_translation_fidelity(source_record: dict[str, Any], payload: dict[str, Any]) -> None:
    """MathDial翻訳が重要な数値・数式tokenを失っていないことを検証する。"""
    errors = mathdial_translation_fidelity_errors(source_record, payload)
    if errors:
        details = ", ".join(f"{field}={tokens[:10]}" for field, tokens in errors.items())
        raise ValueError(f"MathDial翻訳で数値・数式tokenが失われました: {details}")


def meditod_translation_fidelity_errors(
    source_record: dict[str, Any], payload: dict[str, Any]
) -> dict[str, list[str]]:
    """医療相談翻訳の数値・否定・保護医学語の欠落を検査する。"""
    source_fields = {
        "prompt": str(source_record.get("prompt", "")),
        "chosen": str(source_record.get("response", "")),
    }
    translated_fields = {
        "prompt": str(payload["translated_prompt"]),
        "chosen": str(payload["translated_chosen"]),
    }
    errors: dict[str, list[str]] = {}
    english_negation = re.compile(r"\b(?:no|not|never|without|denies?|negative for|doesn't|don't|isn't|hasn't|haven't)\b", re.I)
    japanese_negation = re.compile(
        r"(?:ない|なく|なかっ|ません|いません|ず|否定|認め(?:ない|ません)|なし|no|not|never|without)",
        re.I,
    )
    for field in source_fields:
        missing_numbers = missing_meditod_numeric_tokens(
            source_fields[field],
            translated_fields[field],
        )
        if missing_numbers:
            errors[f"{field}_numbers"] = missing_numbers
        if english_negation.search(source_fields[field]) and not japanese_negation.search(translated_fields[field]):
            errors[f"{field}_negation"] = ["negation"]
    protected = [
        str(value).strip()
        for value in source_record.get("metadata", {}).get("protected_medical_terms", [])
        if str(value).strip()
    ]
    combined_translation = f"{translated_fields['prompt']} {translated_fields['chosen']}".casefold()
    missing_terms = [term for term in protected if term.casefold() not in combined_translation]
    if missing_terms:
        errors["protected_medical_terms"] = missing_terms
    from tools.wildchat_health import has_explicit_unsafe_medical_advice

    if any(
        has_explicit_unsafe_medical_advice(str(value))
        for value in (
            payload.get("translated_chosen", ""),
            *payload.get("rejected_candidates", []),
        )
    ):
        errors["explicit_unsafe_medical_advice"] = ["generated_response"]
    return errors


def validate_meditod_translation_fidelity(source_record: dict[str, Any], payload: dict[str, Any]) -> None:
    errors = meditod_translation_fidelity_errors(source_record, payload)
    if errors:
        raise ValueError(f"MediTOD翻訳で医療情報が失われました: {errors}")


def retry_meditod_translation_for_fidelity(
    *,
    source_record: dict[str, Any],
    payload: dict[str, Any],
    index: int,
    generator: TextGenerator,
    instructions: str,
    model: str,
    max_output_tokens: int,
    candidates: int,
    seed: int,
) -> dict[str, Any]:
    repaired = dict(payload)
    for attempt in range(1, 3):
        errors = meditod_translation_fidelity_errors(source_record, repaired)
        repairable = {
            key: value
            for key, value in errors.items()
            if key != "explicit_unsafe_medical_advice"
        }
        if not repairable:
            validate_meditod_translation_fidelity(source_record, repaired)
            break
        retry_instructions = (
            "あなたは医療相談データの翻訳校正者です。JSONのみを返してください。"
            "translated_promptとtranslated_chosenを、対応する英語原文へ忠実になるよう修正します。"
            "否定、発症時期、期間、数値、単位、薬剤名、症状名を省略・反転せず、"
            "原文にない診断や助言は追加しないでください。論文の引用番号は再挿入不要です。"
            "出力にはtranslated_promptとtranslated_chosenだけを含めてください。"
        )
        repair_input = (
            f"attempt: {attempt}\n"
            f"欠落判定: {json.dumps(repairable, ensure_ascii=False)}\n\n"
            f"english_prompt:\n{source_record.get('prompt', '')}\n\n"
            f"current_translated_prompt:\n{repaired.get('translated_prompt', '')}\n\n"
            f"english_chosen_response:\n{source_record.get('response', '')}\n\n"
            f"current_translated_chosen:\n{repaired.get('translated_chosen', '')}"
        )
        output_text = generator.generate(
            instructions=retry_instructions,
            input_text=repair_input,
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
        patch = extract_json_object(output_text)
        translated_prompt = str(patch.get("translated_prompt", "")).strip()
        translated_chosen = str(patch.get("translated_chosen", "")).strip()
        if not translated_prompt or not translated_chosen:
            raise ValueError("MediTOD医療情報保持の修復JSONに翻訳本文がありません。")
        repaired["translated_prompt"] = translated_prompt
        repaired["translated_chosen"] = translated_chosen
        try:
            validate_meditod_translation_fidelity(source_record, repaired)
            break
        except ValueError:
            if attempt == 2:
                raise
    repaired["generation_retry"] = "meditod_medical_fidelity_targeted_retry"
    repaired["medical_fidelity_version"] = MEDITOD_MEDICAL_FIDELITY_VERSION
    return repaired


def retry_mathdial_translation_for_numeric_fidelity(
    *,
    source_record: dict[str, Any],
    payload: dict[str, Any],
    index: int,
    generator: TextGenerator,
    instructions: str,
    model: str,
    max_output_tokens: int,
    candidates: int,
    seed: int,
) -> dict[str, Any]:
    """数値欠落時だけ、保持すべきtokenを明示して翻訳生成を1回やり直す。"""
    errors = mathdial_translation_fidelity_errors(source_record, payload)
    missing = [token for tokens in errors.values() for token in tokens]
    retry_instructions = (
        instructions
        + "\n\n数値保持の再試行です。english_promptとenglish_chosen_responseの数値、"
        "小数、分数、割合、単位を省略・要約・丸めず、対応する翻訳フィールド内へ保持してください。"
        "自然な桁区切りの追加は許可します。会話内容や指導戦略は変更しないでください。"
        f"\n欠落判定token: {json.dumps(errors, ensure_ascii=False)}"
    )
    retry_payload, skip_reason = generate_translation_payload(
        source_record=source_record,
        index=index,
        generator=generator,
        instructions=retry_instructions,
        model=model,
        max_output_tokens=max_output_tokens,
        candidates=candidates,
        seed=seed + 1_000_003,
    )
    if retry_payload is None:
        raise ValueError(
            f"MathDial数値保持の再翻訳に失敗しました: reason={skip_reason}, missing={missing[:10]}"
        )
    validate_mathdial_translation_fidelity(source_record, retry_payload)
    retry_payload["generation_retry"] = "mathdial_numeric_fidelity_retry"
    retry_payload["numeric_fidelity_version"] = MATHDIAL_NUMERIC_FIDELITY_VERSION
    return retry_payload


def score_japanese_response(
    *,
    record: dict[str, Any],
    response: str,
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    score_model: str,
    max_output_tokens: int,
    style_preset: str = DEFAULT_STYLE_PRESET,
) -> dict[str, Any]:
    """日本語prompt/responseを観測ラベル化し、ベイズスコアを計算する。"""
    scoring_record = {
        "conversation_id": record["conversation_id"],
        "turn_index": record["turn_index"],
        "prompt": record["prompt"],
        "response": response,
    }
    scoring_preset = {
        "mathdial_tutoring": "mathdial_tutoring",
        "meditod_history_taking": "meditod_history_taking",
    }.get(style_preset, "legacy")
    return score_single_record(
        scoring_record,
        bayes_model=bayes_model,
        generator=generator,
        model=score_model,
        max_output_tokens=max_output_tokens,
        instructions=build_transition_scoring_instructions(
            bayes_model, scoring_preset=scoring_preset
        ),
        prior_distribution=record.get("prior_state_distribution"),
        progress_label="[STEP 5/6] japanese rescore",
        scoring_preset=scoring_preset,
        invalid_observation_retries=2 if scoring_preset != "legacy" else 0,
    )


def generate_translation_payload(
    *,
    source_record: dict[str, Any],
    index: int,
    generator: TextGenerator,
    instructions: str,
    model: str,
    max_output_tokens: int,
    candidates: int,
    seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """翻訳・rejected生成を行う。LLM由来の失敗はサンプル単位のskip理由へ変換する。"""
    try:
        output_text = generator.generate(
            instructions=instructions,
            input_text=build_translation_rejected_input(source_record, candidates=candidates, seed=seed + index),
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
    except Exception as exc:
        if not is_content_filter_error(exc):
            raise
        print(
            "[STEP 5/6] content_filter generation retry with sanitized input "
            f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
            flush=True,
        )
        try:
            output_text = generator.generate(
                instructions=instructions,
                input_text=build_safe_translation_rejected_input(
                    source_record,
                    candidates=candidates,
                    seed=seed + index,
                ),
                model=model,
                max_output_tokens=max_output_tokens,
                response_text_format={"type": "json_object"},
            )
            payload = validate_translation_payload(extract_json_object(output_text), candidates=candidates)
            payload["generation_retry"] = "content_filter_sanitized_retry"
            return payload, None
        except ValueError as retry_exc:
            print(
                "[STEP 5/6] skip invalid generation after content_filter retry "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: {retry_exc}",
                flush=True,
            )
            return None, "invalid_generation"
        except Exception as retry_exc:
            if not is_content_filter_error(retry_exc):
                raise
            print(
                "[STEP 5/6] skip content_filter generation "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
                flush=True,
            )
            return None, "content_filter_generation"
    try:
        return validate_translation_payload(extract_json_object(output_text), candidates=candidates), None
    except ValueError as exc:
        print(
            "[STEP 5/6] skip invalid generation "
            f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: {exc}",
            flush=True,
        )
        return None, "invalid_generation"


def choose_rejected(
    *,
    base_record: dict[str, Any],
    rejected_candidates: list[str],
    chosen_score: dict[str, Any],
    bayes_model: TransitionBayesModel,
    generator: TextGenerator,
    score_model: str,
    max_output_tokens: int,
    style_preset: str = DEFAULT_STYLE_PRESET,
) -> tuple[str, dict[str, Any], float]:
    """複数rejected候補を再スコアリングし、score_gap最大の候補を選ぶ。"""
    best_text = ""
    best_score: dict[str, Any] | None = None
    best_gap = -1.0
    chosen_posterior = float(chosen_score["posterior"])
    for candidate in rejected_candidates:
        if candidate == base_record["response"]:
            continue
        rejected_score = score_japanese_response(
            record=base_record,
            response=candidate,
            bayes_model=bayes_model,
            generator=generator,
            score_model=score_model,
            max_output_tokens=max_output_tokens,
            style_preset=style_preset,
        )
        gap = chosen_posterior - float(rejected_score["posterior"])
        if gap > best_gap:
            best_text = candidate
            best_score = rejected_score
            best_gap = gap
    if best_score is None:
        raise ValueError("採用可能なrejected候補がありません。")
    return best_text, best_score, best_gap


def passes_thresholds(
    record: dict[str, Any],
    *,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
    gap_rescue_max_rejected_posterior: float | None = None,
    gap_rescue_min_score_gap: float | None = None,
) -> str | None:
    """DPO候補が採用しきい値を満たす場合、採用ルール名を返す。"""
    chosen = float(record.get("score_chosen", -1.0))
    rejected = float(record.get("score_rejected", 1.0))
    gap = float(record.get("score_gap", -1.0))
    if chosen < min_chosen_posterior:
        return None
    if rejected <= max_rejected_posterior and gap >= min_score_gap:
        return "strict"
    if (
        gap_rescue_max_rejected_posterior is not None
        and gap_rescue_min_score_gap is not None
        and rejected <= gap_rescue_max_rejected_posterior
        and gap >= gap_rescue_min_score_gap
    ):
        return "gap_rescue"
    return None


def skip_record_from_candidate(
    candidate: dict[str, Any],
    *,
    skip_reason: str,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
    gap_rescue_max_rejected_posterior: float | None = None,
    gap_rescue_min_score_gap: float | None = None,
) -> dict[str, Any]:
    """後からしきい値だけ変えて再採用できるskip詳細レコードを作る。"""
    record = dict(candidate)
    record["skip_reason"] = skip_reason
    record["skipped_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    record["acceptance_thresholds"] = {
        "min_score_gap": min_score_gap,
        "min_chosen_posterior": min_chosen_posterior,
        "max_rejected_posterior": max_rejected_posterior,
        "gap_rescue_max_rejected_posterior": gap_rescue_max_rejected_posterior,
        "gap_rescue_min_score_gap": gap_rescue_min_score_gap,
    }
    record["would_accept_with_current_thresholds"] = False
    return record


def build_one_dpo_record(
    source_record: dict[str, Any],
    *,
    index: int,
    bayes_model: TransitionBayesModel,
    model_version: str,
    generator: TextGenerator,
    instructions: str,
    model: str,
    score_model: str,
    max_output_tokens: int,
    candidates: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
    gap_rescue_max_rejected_posterior: float | None,
    gap_rescue_min_score_gap: float | None,
    seed: int,
    style_preset: str,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """1件の英語候補からDPOレコードを作る。条件未満ならskip理由を返す。"""
    translation_payload, skip_reason = generate_translation_payload(
        source_record=source_record,
        index=index,
        generator=generator,
        instructions=instructions,
        model=model,
        max_output_tokens=max_output_tokens,
        candidates=candidates,
        seed=seed,
    )
    if translation_payload is None:
        return None, skip_reason, None
    if style_preset == "mathdial_tutoring":
        try:
            validate_mathdial_translation_fidelity(source_record, translation_payload)
        except ValueError as exc:
            print(
                "[STEP 5/6] MathDial numeric fidelity retry "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: {exc}",
                flush=True,
            )
            translation_payload = retry_mathdial_translation_for_numeric_fidelity(
                source_record=source_record,
                payload=translation_payload,
                index=index,
                generator=generator,
                instructions=instructions,
                model=model,
                max_output_tokens=max_output_tokens,
                candidates=candidates,
                seed=seed,
            )
        translation_payload.setdefault(
            "numeric_fidelity_version", MATHDIAL_NUMERIC_FIDELITY_VERSION
        )
    elif style_preset == "meditod_history_taking":
        try:
            validate_meditod_translation_fidelity(source_record, translation_payload)
        except ValueError as exc:
            print(
                "[STEP 5/6] MediTOD medical fidelity retry "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: {exc}",
                flush=True,
            )
            translation_payload = retry_meditod_translation_for_fidelity(
                source_record=source_record,
                payload=translation_payload,
                index=index,
                generator=generator,
                instructions=instructions,
                model=model,
                max_output_tokens=max_output_tokens,
                candidates=candidates,
                seed=seed,
            )
        translation_payload.setdefault(
            "medical_fidelity_version", MEDITOD_MEDICAL_FIDELITY_VERSION
        )
    japanese_record = {
        "conversation_id": source_record["conversation_id"],
        "turn_index": source_record["turn_index"],
        "prompt": translation_payload["translated_prompt"],
        "response": translation_payload["translated_chosen"],
        "prior_state_distribution": source_record.get("prior_state_distribution"),
    }
    chosen_score = score_japanese_response(
        record=japanese_record,
        response=translation_payload["translated_chosen"],
        bayes_model=bayes_model,
        generator=generator,
        score_model=score_model,
        max_output_tokens=max_output_tokens,
        style_preset=style_preset,
    )
    rejected_text, rejected_score, score_gap = choose_rejected(
        base_record=japanese_record,
        rejected_candidates=translation_payload["rejected_candidates"],
        chosen_score=chosen_score,
        bayes_model=bayes_model,
        generator=generator,
        score_model=score_model,
        max_output_tokens=max_output_tokens,
        style_preset=style_preset,
    )
    if style_preset == "mathdial_tutoring":
        dpo_prompt = build_mathdial_dpo_prompt_from_context_text(translation_payload["translated_prompt"])
        dpo_prompt_template_version = DPO_PROMPT_TEMPLATE_VERSION
    elif style_preset == "meditod_history_taking":
        dpo_prompt = build_meditod_dpo_prompt_from_context_text(translation_payload["translated_prompt"])
        dpo_prompt_template_version = MEDITOD_DPO_PROMPT_TEMPLATE_VERSION
    else:
        dpo_prompt = build_dpo_prompt_from_context_text(translation_payload["translated_prompt"])
        dpo_prompt_template_version = DPO_PROMPT_TEMPLATE_VERSION
    candidate_record = {
        "prompt": dpo_prompt,
        "chosen": translation_payload["translated_chosen"],
        "rejected": rejected_text,
        "score_chosen": float(chosen_score["posterior"]),
        "score_rejected": float(rejected_score["posterior"]),
        "score_gap": score_gap,
        "source_dataset": source_record.get("metadata", {}).get("source_dataset", "DailyDialog"),
        "source_dialogue_id": source_record.get("conversation_id"),
        "turn_index": source_record.get("turn_index"),
        "history_turns": source_record.get("metadata", {}).get("context_turns"),
        "translated_chosen": translation_payload["translated_chosen"],
        "translated_rejected": rejected_text,
        "state_sequence": [
            {
                "role": "chosen",
                "most_likely_state": chosen_score.get("most_likely_state"),
                "state_posteriors": chosen_score.get("state_posteriors"),
            },
            {
                "role": "rejected",
                "most_likely_state": rejected_score.get("most_likely_state"),
                "state_posteriors": rejected_score.get("state_posteriors"),
            },
        ],
        "strategy_sequence": [
            {
                "role": "chosen",
                "observation": chosen_score.get("observation"),
                "observation_score": chosen_score.get("observation_score"),
            },
            {
                "role": "rejected",
                "observation": rejected_score.get("observation"),
                "observation_score": rejected_score.get("observation_score"),
            },
        ],
        "reward_breakdown": {
            "chosen": chosen_score,
            "rejected": rejected_score,
        },
        "translation_quality_score": translation_payload["translation_quality_score"],
        "raw_translated_prompt": translation_payload["translated_prompt"],
        "generation_retry": translation_payload.get("generation_retry"),
        "numeric_fidelity_version": translation_payload.get("numeric_fidelity_version"),
        "medical_fidelity_version": translation_payload.get("medical_fidelity_version"),
        "model_used_for_scoring": score_model,
        "model_used_for_translation": model,
        "model_used_for_rejected_generation": model,
        "bayesian_model_version": model_version,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "dpo_prompt_template_version": dpo_prompt_template_version,
        "source_prompt_en": source_record.get("prompt"),
        "source_chosen_en": source_record.get("response"),
        "metadata": {
            "source_dataset": source_record.get("metadata", {}).get("source_dataset", "DailyDialog"),
            "source_split": source_record.get("metadata", {}).get("source_split"),
            "history_turns": source_record.get("metadata", {}).get("context_turns"),
            "source_strategy": source_record.get("metadata", {}).get("strategy"),
            "raw_translated_prompt": translation_payload["translated_prompt"],
            "source_posterior_en": source_record.get("posterior"),
            "generation_model": model,
            "scoring_model": score_model,
            "seed": seed,
            "bayes_model_name": bayes_model.name,
            "bayes_model_version": model_version,
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "dpo_prompt_template": dpo_prompt_template_version,
            "style_preset": style_preset,
            "rejected_candidates": candidates,
            "generation_retry": translation_payload.get("generation_retry"),
            "numeric_fidelity_version": translation_payload.get("numeric_fidelity_version"),
            "medical_fidelity_version": translation_payload.get("medical_fidelity_version"),
            "source_prompt_hash": hashlib.sha256(str(source_record.get("prompt", "")).encode("utf-8")).hexdigest(),
            "translated_prompt_hash": hashlib.sha256(translation_payload["translated_prompt"].encode("utf-8")).hexdigest(),
            "rejected_prompt_hash": hashlib.sha256(translation_payload["translated_prompt"].encode("utf-8")).hexdigest(),
            "gold": bool(source_record.get("metadata", {}).get("gold", False)),
        },
    }
    if float(chosen_score["posterior"]) < min_chosen_posterior:
        return None, "low_chosen", skip_record_from_candidate(
            candidate_record,
            skip_reason="low_chosen",
            min_score_gap=min_score_gap,
            min_chosen_posterior=min_chosen_posterior,
            max_rejected_posterior=max_rejected_posterior,
            gap_rescue_max_rejected_posterior=gap_rescue_max_rejected_posterior,
            gap_rescue_min_score_gap=gap_rescue_min_score_gap,
        )
    acceptance_rule = passes_thresholds(
        candidate_record,
        min_score_gap=min_score_gap,
        min_chosen_posterior=min_chosen_posterior,
        max_rejected_posterior=max_rejected_posterior,
        gap_rescue_max_rejected_posterior=gap_rescue_max_rejected_posterior,
        gap_rescue_min_score_gap=gap_rescue_min_score_gap,
    )
    if acceptance_rule is not None:
        candidate_record["acceptance_rule"] = acceptance_rule
        candidate_record["metadata"]["acceptance_rule"] = acceptance_rule
        return candidate_record, None, None
    if float(rejected_score["posterior"]) > max_rejected_posterior:
        return None, "high_rejected", skip_record_from_candidate(
            candidate_record,
            skip_reason="high_rejected",
            min_score_gap=min_score_gap,
            min_chosen_posterior=min_chosen_posterior,
            max_rejected_posterior=max_rejected_posterior,
            gap_rescue_max_rejected_posterior=gap_rescue_max_rejected_posterior,
            gap_rescue_min_score_gap=gap_rescue_min_score_gap,
        )
    return None, "small_gap", skip_record_from_candidate(
        candidate_record,
        skip_reason="small_gap",
        min_score_gap=min_score_gap,
        min_chosen_posterior=min_chosen_posterior,
        max_rejected_posterior=max_rejected_posterior,
        gap_rescue_max_rejected_posterior=gap_rescue_max_rejected_posterior,
        gap_rescue_min_score_gap=gap_rescue_min_score_gap,
    )


def build_dpo_records(
    selected_records: list[dict[str, Any]],
    *,
    bayes_model: TransitionBayesModel,
    bayes_model_path: Path | str,
    generator: TextGenerator,
    model: str,
    score_model: str,
    max_output_tokens: int,
    candidates: int,
    min_score_gap: float,
    min_chosen_posterior: float,
    max_rejected_posterior: float,
    seed: int,
    max_records: int | None,
    gap_rescue_max_rejected_posterior: float | None = None,
    gap_rescue_min_score_gap: float | None = None,
    target_records: int | None = None,
    workers: int = 1,
    style_preset: str = DEFAULT_STYLE_PRESET,
    skip_sample_errors: bool = False,
    output_path: Path | str | None = None,
    skipped_output_path: Path | str | None = None,
    existing_records: list[dict[str, Any]] | None = None,
    existing_skipped_records: list[dict[str, Any]] | None = None,
    audit_log_path: Path | str | None = None,
    stats: DpoGenerationStats | None = None,
    heartbeat_path: Path | str | None = None,
    heartbeat_stage_prefix: str = "dpo_generation",
    max_source_characters: int | None = None,
) -> list[dict[str, Any]]:
    """抽出済み英語レコードから日本語DPOレコードを作る。"""
    dpo_records: list[dict[str, Any]] = list(existing_records or [])
    if stats is not None:
        stats.existing_records = len(existing_records or [])
    skipped_records = list(existing_skipped_records or [])
    instructions = build_translation_rejected_instructions(bayes_model, style_preset=style_preset)
    model_version = bayes_model_version(bayes_model_path)
    source_records = selected_records[:max_records] if max_records is not None else selected_records
    over_length_records = [
        record
        for record in source_records
        if max_source_characters is not None
        and len(str(record.get("prompt", ""))) + len(str(record.get("response", "")))
        > max_source_characters
    ]
    if over_length_records:
        print(
            f"[STEP 5/6] exclude over-length sources before API: "
            f"count={len(over_length_records)} max_source_characters={max_source_characters}",
            flush=True,
        )
        if stats is not None:
            stats.skipped_source_too_long += len(over_length_records)
    source_records = [
        record
        for record in source_records
        if max_source_characters is None
        or len(str(record.get("prompt", ""))) + len(str(record.get("response", "")))
        <= max_source_characters
    ]
    done_keys = {dpo_record_key(record) for record in dpo_records}
    promoted_from_skipped = 0
    for skipped_record in skipped_records:
        key = dpo_record_key(skipped_record)
        if key in done_keys:
            continue
        acceptance_rule = passes_thresholds(
            skipped_record,
            min_score_gap=min_score_gap,
            min_chosen_posterior=min_chosen_posterior,
            max_rejected_posterior=max_rejected_posterior,
            gap_rescue_max_rejected_posterior=gap_rescue_max_rejected_posterior,
            gap_rescue_min_score_gap=gap_rescue_min_score_gap,
        )
        if acceptance_rule is None:
            continue
        promoted_record = dpo_record_without_skip_admin(skipped_record)
        promoted_record["acceptance_rule"] = acceptance_rule
        promoted_record.setdefault("metadata", {})["acceptance_rule"] = acceptance_rule
        dpo_records.append(promoted_record)
        done_keys.add(key)
        promoted_from_skipped += 1
        if output_path is not None:
            append_jsonl(promoted_record, output_path)
        if target_records is not None and len(dpo_records) >= target_records:
            break
    if promoted_from_skipped:
        print(
            f"[STEP 5/6] promoted skipped candidates: count={promoted_from_skipped} "
            f"max_rejected_posterior={max_rejected_posterior} "
            f"gap_rescue_max_rejected_posterior={gap_rescue_max_rejected_posterior} "
            f"gap_rescue_min_score_gap={gap_rescue_min_score_gap}",
            flush=True,
        )
    skipped_keys = {
        dpo_record_key(record)
        for record in skipped_records
        if "source_dialogue_id" in record and "turn_index" in record
    }
    processed_keys = done_keys | skipped_keys
    pending_records = [
        record
        for record in source_records
        if source_record_key(record) not in processed_keys
    ]
    if done_keys:
        print(f"[STEP 5/6] dpo generation resume: skipped={len(done_keys)} pending={len(pending_records)}", flush=True)
    if target_records is not None and len(dpo_records) >= target_records:
        print(
            f"[STEP 5/6] target already reached: accepted={len(dpo_records)} target={target_records}",
            flush=True,
        )
        return sorted(dpo_records, key=lambda record: record["score_gap"], reverse=True)[:target_records]
    skipped_low_chosen = 0
    skipped_high_rejected = 0
    skipped_small_gap = 0
    skipped_content_filter_generation = 0
    skipped_invalid_generation = 0
    skipped_sample_error = 0
    indexed_records = list(enumerate(pending_records, start=1))

    def update_heartbeat(completed: int) -> None:
        skipped_total = (
            skipped_low_chosen
            + skipped_high_rejected
            + skipped_small_gap
            + skipped_content_filter_generation
            + skipped_invalid_generation
            + skipped_sample_error
        )
        write_heartbeat(
            heartbeat_path,
            f"{heartbeat_stage_prefix}_running",
            {
                "completed": completed,
                "pending": len(indexed_records),
                "accepted": len(dpo_records),
                "skipped": skipped_total,
            },
        )

    update_heartbeat(0)

    def handle_result(
        record: dict[str, Any] | None,
        skip_reason: str | None,
        skip_record: dict[str, Any] | None,
    ) -> None:
        nonlocal skipped_low_chosen, skipped_high_rejected, skipped_small_gap
        nonlocal skipped_content_filter_generation, skipped_invalid_generation, skipped_sample_error
        if skip_record is not None and skipped_output_path is not None:
            append_jsonl(skip_record, skipped_output_path)
        if skip_reason == "low_chosen":
            skipped_low_chosen += 1
            if stats is not None:
                stats.skipped_low_chosen += 1
            print(
                "[STEP 5/6] skip low chosen posterior "
                f"chosen={float(skip_record['score_chosen']):.3f} "
                f"rejected={float(skip_record['score_rejected']):.3f} "
                f"gap={float(skip_record['score_gap']):.3f}"
                if skip_record is not None
                else "[STEP 5/6] skip low chosen posterior",
                flush=True,
            )
            return
        if skip_reason == "high_rejected":
            skipped_high_rejected += 1
            if stats is not None:
                stats.skipped_high_rejected += 1
            print(
                "[STEP 5/6] skip high rejected posterior "
                f"chosen={float(skip_record['score_chosen']):.3f} "
                f"rejected={float(skip_record['score_rejected']):.3f} "
                f"gap={float(skip_record['score_gap']):.3f}"
                if skip_record is not None
                else "[STEP 5/6] skip high rejected posterior",
                flush=True,
            )
            return
        if skip_reason == "small_gap":
            skipped_small_gap += 1
            if stats is not None:
                stats.skipped_small_gap += 1
            print(
                "[STEP 5/6] skip small score_gap "
                f"chosen={float(skip_record['score_chosen']):.3f} "
                f"rejected={float(skip_record['score_rejected']):.3f} "
                f"gap={float(skip_record['score_gap']):.3f}"
                if skip_record is not None
                else "[STEP 5/6] skip small score_gap",
                flush=True,
            )
            return
        if skip_reason == "content_filter_generation":
            skipped_content_filter_generation += 1
            if stats is not None:
                stats.skipped_content_filter_generation += 1
            print("[STEP 5/6] skip content_filter generation", flush=True)
            return
        if skip_reason == "invalid_generation":
            skipped_invalid_generation += 1
            if stats is not None:
                stats.skipped_invalid_generation += 1
            print("[STEP 5/6] skip invalid generation", flush=True)
            return
        if skip_reason == "sample_error":
            skipped_sample_error += 1
            if stats is not None:
                stats.skipped_sample_error += 1
            print("[STEP 5/6] skip sample error", flush=True)
            return
        if record is None:
            return
        if target_records is not None and len(dpo_records) >= target_records:
            return
        dpo_records.append(record)
        if stats is not None:
            stats.accepted_new += 1
        if output_path is not None:
            append_jsonl(record, output_path)
        print(
            f"[STEP 5/6] accepted score_gap={float(record['score_gap']):.3f} "
            f"chosen={float(record['score_chosen']):.3f} "
            f"rejected={float(record['score_rejected']):.3f} "
            f"rule={record.get('acceptance_rule', 'strict')}",
            flush=True,
        )

    def build_record_safely(
        index: int,
        source_record: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
        try:
            return build_one_dpo_record(
                source_record,
                index=index,
                bayes_model=bayes_model,
                model_version=model_version,
                generator=generator,
                instructions=instructions,
                model=model,
                score_model=score_model,
                max_output_tokens=max_output_tokens,
                candidates=candidates,
                min_score_gap=min_score_gap,
                min_chosen_posterior=min_chosen_posterior,
                max_rejected_posterior=max_rejected_posterior,
                gap_rescue_max_rejected_posterior=gap_rescue_max_rejected_posterior,
                gap_rescue_min_score_gap=gap_rescue_min_score_gap,
                seed=seed,
                style_preset=style_preset,
            )
        except Exception as exc:
            if not skip_sample_errors:
                raise
            print(
                "[STEP 5/6] skip sample error "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            error_record = {
                "source_dialogue_id": source_record.get("conversation_id"),
                "turn_index": source_record.get("turn_index"),
                "skip_reason": "sample_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "source_prompt_hash": hashlib.sha256(
                    str(source_record.get("prompt", "")).encode("utf-8")
                ).hexdigest(),
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "numeric_fidelity_version": (
                    MATHDIAL_NUMERIC_FIDELITY_VERSION
                    if style_preset == "mathdial_tutoring"
                    else None
                ),
                "medical_fidelity_version": (
                    MEDITOD_MEDICAL_FIDELITY_VERSION
                    if style_preset == "meditod_history_taking"
                    else None
                ),
                "skipped_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            return None, "sample_error", error_record

    if workers <= 1:
        for index, source_record in indexed_records:
            if target_records is not None and len(dpo_records) >= target_records:
                print(f"[STEP 5/6] target reached: accepted={len(dpo_records)} target={target_records}", flush=True)
                break
            print(
                f"[STEP 5/6] dpo generation: {index}/{len(pending_records)} "
                f"accepted={len(dpo_records)} "
                f"skipped={skipped_low_chosen + skipped_high_rejected + skipped_small_gap + skipped_content_filter_generation + skipped_invalid_generation + skipped_sample_error} "
                f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
                flush=True,
            )
            if stats is not None:
                stats.attempted += 1
            record, skip_reason, skip_record = build_record_safely(index, source_record)
            handle_result(record, skip_reason, skip_record)
            update_heartbeat(index)
    else:
        print(f"[STEP 5/6] dpo generation parallel workers={workers} pending={len(indexed_records)}", flush=True)
        completed = 0
        for chunk_start in range(0, len(indexed_records), workers):
            if target_records is not None and len(dpo_records) >= target_records:
                print(f"[STEP 5/6] target reached: accepted={len(dpo_records)} target={target_records}", flush=True)
                break
            chunk = indexed_records[chunk_start : chunk_start + workers]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        build_record_safely,
                        index,
                        source_record,
                    ): source_record
                    for index, source_record in chunk
                }
                for future in as_completed(futures):
                    source_record = futures[future]
                    record, skip_reason, skip_record = future.result()
                    completed += 1
                    if stats is not None:
                        stats.attempted += 1
                    progress = completed / len(indexed_records) * 100.0 if indexed_records else 100.0
                    print(
                        f"[STEP 5/6] dpo generation: {completed}/{len(indexed_records)} "
                        f"({progress:.1f}%) accepted={len(dpo_records)} "
                        f"skipped={skipped_low_chosen + skipped_high_rejected + skipped_small_gap + skipped_content_filter_generation + skipped_invalid_generation + skipped_sample_error} "
                        f"{source_record.get('conversation_id')}#{source_record.get('turn_index')}",
                        flush=True,
                    )
                    handle_result(record, skip_reason, skip_record)
                    update_heartbeat(completed)
    write_heartbeat(
        heartbeat_path,
        f"{heartbeat_stage_prefix}_complete",
        {
            "completed": len(indexed_records),
            "pending": len(indexed_records),
            "accepted": len(dpo_records),
            "skipped": (
                skipped_low_chosen
                + skipped_high_rejected
                + skipped_small_gap
                + skipped_content_filter_generation
                + skipped_invalid_generation
                + skipped_sample_error
            ),
        },
    )
    print(
        f"[STEP 5/6] dpo generation complete: accepted={len(dpo_records)} "
        f"skipped_low_chosen={skipped_low_chosen} "
        f"skipped_high_rejected={skipped_high_rejected} "
        f"skipped_small_gap={skipped_small_gap} "
        f"skipped_content_filter_generation={skipped_content_filter_generation} "
        f"skipped_invalid_generation={skipped_invalid_generation} "
        f"skipped_sample_error={skipped_sample_error}",
        flush=True,
    )
    if audit_log_path is not None:
        append_audit_log(
            title="日本語DPO preferenceデータ生成",
            target_files=[
                str(bayes_model_path),
                str(output_path) if output_path is not None else "(memory only)",
            ],
            operation="抽出済み高スコア英語応答を自然な日本語chosenへ翻訳し、自然だが低スコアなrejected候補を生成・再スコアリングした。",
            reason="QwenのDPO/LoRA学習に使うprompt/chosen/rejected形式の研究データを作成するため。",
            alternatives=[
                "rejectedを単純な悪文にする案はDPO品質が落ちるため採用しなかった。",
                "翻訳後chosenを再スコアリングしない案は、翻訳で会話戦略が崩れたサンプルを検出できないため採用しなかった。",
            ],
            command=(
                "python3 -m tools.translate_and_generate_dpo "
                f"--bayes-model {bayes_model_path} "
                f"--model {model} --score-model {score_model} "
                f"--candidates {candidates} --min-score-gap {min_score_gap} "
                f"--min-chosen-posterior {min_chosen_posterior} "
                f"--max-rejected-posterior {max_rejected_posterior} "
                f"--gap-rescue-max-rejected-posterior {gap_rescue_max_rejected_posterior} "
                f"--gap-rescue-min-score-gap {gap_rescue_min_score_gap} "
                f"--workers {workers} --seed {seed}"
            ),
            before_after=[
                f"入力候補数: {len(source_records)}",
                f"既存採用済み件数: {len(existing_records or [])}",
                f"最終採用件数: {len(dpo_records)}",
                f"target_records: {target_records}",
                f"skip low chosen posterior: {skipped_low_chosen}",
                f"skip high rejected posterior: {skipped_high_rejected}",
                f"skip small score_gap: {skipped_small_gap}",
                f"skip content_filter generation: {skipped_content_filter_generation}",
                f"skip invalid generation: {skipped_invalid_generation}",
                f"skip sample error: {skipped_sample_error}",
                f"skip source too long before API: {len(over_length_records)}",
            ],
            risks=[
                "content_filter_generationは該当サンプルをDPOデータから除外するため、件数不足時は候補数を増やす必要がある。",
                "skip low chosen posteriorが多い場合、翻訳または抽出条件が小コーパス由来の戦略を十分保持していない可能性がある。",
                "skip small score_gapが多い場合、chosen/rejectedの対比が弱くDPO効果が出にくい可能性がある。",
            ],
            audit_log_path=audit_log_path,
            details=[
                f"generation_model: {model}",
                f"score_model: {score_model}",
                f"prompt_template_version: {PROMPT_TEMPLATE_VERSION}",
                f"dpo_prompt_template_version: {DPO_PROMPT_TEMPLATE_VERSION}",
                f"style_preset: {style_preset}",
                f"bayes_model_version: {model_version}",
                f"gap_rescue_max_rejected_posterior: {gap_rescue_max_rejected_posterior}",
                f"gap_rescue_min_score_gap: {gap_rescue_min_score_gap}",
            ],
        )
    sorted_records = sorted(dpo_records, key=lambda record: record["score_gap"], reverse=True)
    if target_records is not None:
        return sorted_records[:target_records]
    return sorted_records


def main() -> int:
    """CLIエントリポイント。"""
    args = parse_args()
    selected_records = read_jsonl(args.input)
    bayes_model = load_transition_bayes_model(args.bayes_model)
    if args.dry_run:
        print("日本語DPO生成 dry-run")
        print(f"  input_records: {len(selected_records)}")
        print(f"  bayes_model: {bayes_model.name}")
        print(f"  generation_model: {args.model}")
        print(f"  score_model: {args.score_model}")
        return 0
    stats = DpoGenerationStats()
    skipped_output = args.skipped_output or str(default_skipped_output_path(args.output))
    dpo_records = build_dpo_records(
        selected_records,
        bayes_model=bayes_model,
        bayes_model_path=args.bayes_model,
        generator=OpenAIResponsesGenerator(),
        model=args.model,
        score_model=args.score_model,
        max_output_tokens=args.max_output_tokens,
        candidates=args.candidates,
        min_score_gap=args.min_score_gap,
        min_chosen_posterior=args.min_chosen_posterior,
        max_rejected_posterior=args.max_rejected_posterior,
        seed=args.seed,
        max_records=args.max_records,
        gap_rescue_max_rejected_posterior=args.gap_rescue_max_rejected_posterior,
        gap_rescue_min_score_gap=args.gap_rescue_min_score_gap,
        target_records=args.target_records,
        workers=max(1, args.workers),
        style_preset=args.style_preset,
        skip_sample_errors=args.skip_sample_errors,
        output_path=args.output,
        skipped_output_path=skipped_output,
        existing_records=read_existing_dpo_records(args.output),
        existing_skipped_records=read_existing_skipped_records(skipped_output),
        audit_log_path=args.audit_log,
        stats=stats,
        heartbeat_path=args.heartbeat_file,
        heartbeat_stage_prefix=args.heartbeat_stage_prefix,
        max_source_characters=args.max_source_characters,
    )
    if (
        args.target_records is not None
        and len(dpo_records) < args.target_records
        and not args.allow_target_shortfall
    ):
        raise RuntimeError(
            f"DPO採用件数がtarget_recordsへ届きませんでした: "
            f"accepted={len(dpo_records)}, target={args.target_records}"
        )
    if args.target_records is not None and len(dpo_records) < args.target_records:
        print(
            "[STEP 5/6] target shortfall retained for candidate expansion: "
            f"accepted={len(dpo_records)} target={args.target_records}",
            flush=True,
        )
    write_jsonl(dpo_records, args.output)
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    write_json(
        {
            "input": args.input,
            "output": args.output,
            "skipped_output": skipped_output,
            "bayes_model": args.bayes_model,
            "bayes_model_version": bayes_model_version(args.bayes_model),
            "generation_model": args.model,
            "audit_model": args.audit_model,
            "score_model": args.score_model,
            "seed": args.seed,
            "prompt_template": PROMPT_TEMPLATE_VERSION,
            "dpo_prompt_template": DPO_PROMPT_TEMPLATE_VERSION,
            "style_preset": args.style_preset,
            "skip_sample_errors": args.skip_sample_errors,
            "min_score_gap": args.min_score_gap,
            "min_chosen_posterior": args.min_chosen_posterior,
            "max_rejected_posterior": args.max_rejected_posterior,
            "gap_rescue_max_rejected_posterior": args.gap_rescue_max_rejected_posterior,
            "gap_rescue_min_score_gap": args.gap_rescue_min_score_gap,
            "generation_stats": stats.as_dict(),
            "max_output_tokens": args.max_output_tokens,
            "max_source_characters": args.max_source_characters,
            "source_length_policy": "exclude_whole_sample_without_truncating_history",
            "workers": max(1, args.workers),
            "target_records": args.target_records,
            "records_written": len(dpo_records),
        },
        manifest_path,
    )
    print(f"日本語DPO JSONLを書き出しました: {args.output} ({len(dpo_records)} 件)")
    print(f"skip詳細JSONL: {skipped_output}")
    print(f"再現性manifestを書き出しました: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
