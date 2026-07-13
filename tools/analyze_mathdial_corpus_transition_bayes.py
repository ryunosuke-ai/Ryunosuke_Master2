"""MathDial会話とTeacher moveから遷移ベイズモデルを直接生成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from core.transition_bayes_model import parse_transition_bayes_model
from tools.analyze_small_corpus import (
    OpenAIResponsesGenerator,
    TextGenerator,
    extract_json_object,
    load_env_file,
    resolve_analysis_model,
)
from tools.analyze_small_corpus_transition_bayes import build_json_repair_instructions


DEFAULT_MAX_OUTPUT_TOKENS = 24_000
DEFAULT_MAX_INPUT_CHARS = 300_000
DEFAULT_EMISSION_MARGIN = 0.10
DEFAULT_MIN_NEGATIVE_OBSERVATIONS = 2


class MathDialModelQualityError(ValueError):
    """生成候補がMathDial固有の識別性要件を満たさない場合の例外。"""

    def __init__(self, message: str, *, candidate: dict[str, Any], diagnostics: dict[str, Any]):
        super().__init__(message)
        self.candidate = candidate
        self.diagnostics = diagnostics


def read_analysis_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}をJSONとして読めません: {exc}") from exc
            if record.get("source_split") != "train":
                raise ValueError(f"分析標本へtrain以外が混入しています: {record.get('conversation_id')}")
            if not record.get("qid") or not isinstance(record.get("dialog"), list):
                raise ValueError(f"分析標本schemaが不正です: {record.get('conversation_id')}")
            records.append(record)
    if not records:
        raise ValueError("MathDial分析標本が空です。")
    if len({row["qid"] for row in records}) != len(records):
        raise ValueError("MathDial分析標本内でqidが重複しています。")
    return records


def build_mathdial_corpus_text(
    records: list[dict[str, Any]], *, max_chars: int = DEFAULT_MAX_INPUT_CHARS
) -> str:
    lines: list[str] = []
    for record in sorted(records, key=lambda row: str(row["conversation_id"])):
        lines.extend([
            f"\n# conversation_id={record['conversation_id']}",
            "## task_annotations",
            f"qid: {record['qid']}",
            f"question: {record.get('question', '')}",
            f"ground_truth: {record.get('ground_truth', '')}",
            "## dialog",
        ])
        for turn in record["dialog"]:
            annotation = ""
            if turn["speaker"] == "assistant":
                moves = json.dumps(
                    turn.get("annotated_teacher_moves", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                annotation = f" [annotated_teacher_moves={moves}]"
            lines.append(
                f"{int(turn['turn_index'])}. {turn['speaker']}{annotation}: {turn['text']}"
            )
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        raise ValueError(
            "MathDial分析入力が上限を超えています。完全会話を切り詰めずに分析するため、"
            f"--max-input-charsを増やしてください: {len(text)}/{max_chars}"
        )
    return text


def build_mathdial_analysis_instructions() -> str:
    return """あなたは会話コーパス分析、個別指導対話、動的ベイズモデル設計の専門家です。

以下のMathDial形式の小規模会話コーパスを分析し、このコーパスが重視する個別指導の進め方を表す状態遷移ベイズモデルを作成してください。目的は数学問題そのものを抽出することではなく、学習者の誤りや混乱を診断し、質問や段階的ヒントで自己修正を促す会話スタイルを大量対話から選別することです。

利用できる情報:
- question / ground_truth: 問題と参照解答。数学的文脈の確認にだけ使う。
- dialog: assistant（Teacher）とuser（Student）の完全な複数ターン会話。
- annotated_teacher_moves: MathDial公式のTeacher move。probing、focus、telling、genericがある。連結発話では複数値の場合がある。

Teacher move利用方針:
- annotated_teacher_movesはassistant発話の機能を示す高価値なannotationとして強く参照する。
- ただし4ラベルを機械的に状態名へコピーしない。会話本文、直前の学習者状態、次の学習者反応と照合する。
- probingとfocusが、誤り診断、焦点化、段階的推論、自己修正へどう使い分けられるかを分析する。
- tellingは、診断後の必要な説明・訂正と、情報不足のまま答えを与える早すぎるtellingを区別する。
- genericは、励まし・会話管理として有効な場合と、学習状態に根拠づけられない場合を区別する。
- 診断後に学習者の誤りへ対応して行う説明・訂正は、正の指導戦略として明確に表現する。

モデル設計方針:
1. 状態は会話の進行局面を表す4〜7個とし、単なる数学トピックや表現技法にしない。
2. positive_statesは、診断、足場かけ、自己修正、適切な説明、理解確認などMathDialらしい進行を表す。
3. negative_statesは、早すぎる直接解答、学習者状態を無視した説明、文脈非依存応答などを表す。
4. observationsはprompt/responseから後段LLMが安定分類できるassistant応答戦略を4〜8個作る。
5. initial_state_prior、P(next_state|current_state)、P(observation|state)をコーパス本文とannotationに基づいて設定する。
6. 低頻度・曖昧で区別しにくいラベルは統合する。
7. premature direct answerとは別に、反復、文脈不一致、根拠のない称賛などを表すoff-style observationを必ず作る。
8. 各positive stateと各negative stateに、反対群よりemissionが0.10以上高く、その状態を識別できるobservationを最低1つ持たせる。
9. negative群が優勢なobservationを最低2種類作る。早すぎる直接解答と、文脈不一致・根拠なし応答を同じobservationへ統合しない。
10. stateは潜在的な会話局面、observationは応答から直接分類する機能である。state名とobservation名を同一または機能的に重複させない。

出力制約:
- JSON objectのみを出力し、Markdownや説明文を付けない。
- model_typeはtransition_bayes_network。
- 必須キーはname, model_type, states, positive_states, negative_states, observations, initial_state_prior, transition_likelihoods, emission_likelihoods, state_descriptions, observation_descriptions, dataset_hypothesis。
- statesは4〜7個、observationsは4〜8個。
- ラベルは英小文字・数字・アンダースコアだけを使う。
- positive_statesとnegative_statesはstatesの部分集合で重複させない。
- priorと各確率行は全ラベルを含み、各値は0より大きく1より小さく、行合計を1.0にする。
- descriptionsとdataset_hypothesisは後段の分類に使える具体的な日本語で書く。

以下が分析対象コーパスです。""".strip()


def mock_model() -> dict[str, Any]:
    states = [
        "diagnosing_need",
        "guided_scaffolding",
        "verified_understanding",
        "premature_telling",
        "stalled_misalignment",
    ]
    observations = [
        "elicit_reasoning",
        "focused_hint",
        "diagnosed_explanation",
        "understanding_check",
        "premature_answer",
        "context_misaligned_response",
        "ungrounded_praise",
    ]
    return {
        "name": "mathdial_tutoring_transition_model",
        "model_type": "transition_bayes_network",
        "states": states,
        "positive_states": states[:3],
        "negative_states": states[3:],
        "observations": observations,
        "initial_state_prior": dict(zip(states, [0.40, 0.30, 0.12, 0.10, 0.08])),
        "transition_likelihoods": {
            states[0]: dict(zip(states, [0.25, 0.50, 0.10, 0.08, 0.07])),
            states[1]: dict(zip(states, [0.10, 0.45, 0.32, 0.07, 0.06])),
            states[2]: dict(zip(states, [0.08, 0.17, 0.60, 0.07, 0.08])),
            states[3]: dict(zip(states, [0.12, 0.13, 0.10, 0.55, 0.10])),
            states[4]: dict(zip(states, [0.15, 0.15, 0.10, 0.10, 0.50])),
        },
        "emission_likelihoods": {
            states[0]: dict(zip(observations, [0.50, 0.15, 0.15, 0.08, 0.04, 0.04, 0.04])),
            states[1]: dict(zip(observations, [0.14, 0.43, 0.20, 0.12, 0.03, 0.04, 0.04])),
            states[2]: dict(zip(observations, [0.10, 0.10, 0.22, 0.45, 0.03, 0.04, 0.06])),
            states[3]: dict(zip(observations, [0.08, 0.07, 0.12, 0.06, 0.55, 0.06, 0.06])),
            states[4]: dict(zip(observations, [0.06, 0.06, 0.08, 0.06, 0.06, 0.42, 0.26])),
        },
        "state_descriptions": {
            states[0]: "学習者の試行や誤りを確認して支援方針を定める状態。",
            states[1]: "質問や段階的ヒントで学習者自身の推論を進める状態。",
            states[2]: "自己修正や理解を確認して解決へまとめる状態。",
            states[3]: "十分な診断や足場かけなしに答えを与える状態。",
            states[4]: "学習者の直前発話と対応しない反復や根拠のない称賛で進行が停滞する状態。",
        },
        "observation_descriptions": {
            observations[0]: "学習者の考え方や誤りの理由を尋ねる応答。",
            observations[1]: "答えを明かさず次の一歩へ焦点を当てる質問やヒント。",
            observations[2]: "診断した誤りに対応して必要な説明や訂正を行う応答。",
            observations[3]: "学習者自身に理解や修正結果を確認させる応答。",
            observations[4]: "情報不足のまま最終答えや全手順を提示する応答。",
            observations[5]: "直前の学習者発話と対応しない反復・一般論・別文脈の応答。",
            observations[6]: "学習内容の根拠なしに称賛や会話管理だけを行う応答。",
        },
        "dataset_hypothesis": "学習者状態を診断し、質問と足場かけで自己修正へ導く個別指導対話。",
    }


def evaluate_emission_quality(
    payload: dict[str, Any],
    *,
    margin: float = DEFAULT_EMISSION_MARGIN,
    min_negative_observations: int = DEFAULT_MIN_NEGATIVE_OBSERVATIONS,
) -> dict[str, Any]:
    """正負状態を観測から識別できるか検証し、診断情報を返す。"""
    model = parse_transition_bayes_model(payload)
    positive_rows: dict[str, Any] = {}
    negative_rows: dict[str, Any] = {}
    for state in model.positive_states:
        candidates = {
            observation: model.emission_likelihoods[state][observation]
            - max(model.emission_likelihoods[other][observation] for other in model.negative_states)
            for observation in model.observations
        }
        best = max(candidates, key=candidates.get)
        positive_rows[state] = {
            "best_observation": best,
            "margin": candidates[best],
            "passed": candidates[best] >= margin,
        }
    for state in model.negative_states:
        candidates = {
            observation: model.emission_likelihoods[state][observation]
            - max(model.emission_likelihoods[other][observation] for other in model.positive_states)
            for observation in model.observations
        }
        best = max(candidates, key=candidates.get)
        negative_rows[state] = {
            "best_observation": best,
            "margin": candidates[best],
            "passed": candidates[best] >= margin,
        }
    negative_dominant = []
    for observation in model.observations:
        negative_max = max(
            model.emission_likelihoods[state][observation]
            for state in model.negative_states
        )
        positive_max = max(
            model.emission_likelihoods[state][observation]
            for state in model.positive_states
        )
        if negative_max - positive_max >= margin:
            negative_dominant.append(
                {
                    "observation": observation,
                    "negative_max": negative_max,
                    "positive_max": positive_max,
                    "margin": negative_max - positive_max,
                }
            )
    overlapping_names = sorted(set(model.states) & set(model.observations))
    passed = (
        all(row["passed"] for row in positive_rows.values())
        and all(row["passed"] for row in negative_rows.values())
        and len(negative_dominant) >= min_negative_observations
        and not overlapping_names
    )
    return {
        "passed": passed,
        "required_margin": margin,
        "minimum_negative_dominant_observations": min_negative_observations,
        "positive_state_discriminators": positive_rows,
        "negative_state_discriminators": negative_rows,
        "negative_dominant_observations": negative_dominant,
        "overlapping_state_observation_names": overlapping_names,
    }


def generate_model(
    records: list[dict[str, Any]],
    *,
    generator: TextGenerator | None,
    model: str,
    max_output_tokens: int,
    max_input_chars: int,
    mock: bool,
    emission_margin: float = DEFAULT_EMISSION_MARGIN,
    min_negative_observations: int = DEFAULT_MIN_NEGATIVE_OBSERVATIONS,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], str, str]:
    instructions = build_mathdial_analysis_instructions()
    corpus_text = build_mathdial_corpus_text(records, max_chars=max_input_chars)
    if mock:
        payload = mock_model()
    else:
        assert generator is not None
        if progress:
            progress(f"{model}へMathDial遷移ベイズモデル生成を依頼しています。")
        raw = generator.generate(
            instructions=instructions,
            input_text=corpus_text,
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
        try:
            payload = extract_json_object(raw)
        except ValueError:
            if progress:
                progress("JSON構文修復を依頼しています。")
            repaired = generator.generate(
                instructions=build_json_repair_instructions(),
                input_text=raw,
                model=model,
                max_output_tokens=max_output_tokens,
                response_text_format={"type": "json_object"},
            )
            payload = extract_json_object(repaired)
    try:
        parsed = parse_transition_bayes_model(payload)
        if not 4 <= len(parsed.states) <= 7:
            raise ValueError(f"MathDialモデルのstate数が範囲外です: {len(parsed.states)}")
        if not 4 <= len(parsed.observations) <= 8:
            raise ValueError(f"MathDialモデルのobservation数が範囲外です: {len(parsed.observations)}")
        diagnostics = evaluate_emission_quality(
            payload,
            margin=emission_margin,
            min_negative_observations=min_negative_observations,
        )
        if not diagnostics["passed"]:
            raise ValueError("MathDialモデルのemission識別性gateに不合格です。")
    except ValueError as exc:
        diagnostics = locals().get("diagnostics", {"passed": False, "schema_error": str(exc)})
        raise MathDialModelQualityError(
            str(exc), candidate=payload, diagnostics=diagnostics
        ) from exc
    return payload, instructions, corpus_text


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json_atomic(payload: dict[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def _append_rejected_model(
    path: Path | str,
    *,
    error: Exception,
    model: str,
    candidate: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "analysis_model": model,
        "error": f"{type(error).__name__}: {error}",
        "diagnostics": diagnostics,
        "candidate": candidate,
    }
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="MathDialから遷移ベイズモデルを直接生成")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compat-output")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt-output", required=True)
    parser.add_argument("--input-text-output", required=True)
    parser.add_argument("--quality-report-output", required=True)
    parser.add_argument("--rejected-models-output", required=True)
    parser.add_argument("--model", default=resolve_analysis_model())
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    parser.add_argument("--emission-margin", type=float, default=DEFAULT_EMISSION_MARGIN)
    parser.add_argument(
        "--min-negative-observations",
        type=int,
        default=DEFAULT_MIN_NEGATIVE_OBSERVATIONS,
    )
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()

    def report(message: str) -> None:
        print(f"[{time.monotonic() - started:6.1f}s] {message}", flush=True)

    records = read_analysis_jsonl(args.input)
    try:
        payload, instructions, corpus_text = generate_model(
            records,
            generator=None if args.mock else OpenAIResponsesGenerator(),
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            max_input_chars=args.max_input_chars,
            mock=args.mock,
            emission_margin=args.emission_margin,
            min_negative_observations=args.min_negative_observations,
            progress=report,
        )
    except MathDialModelQualityError as exc:
        _append_rejected_model(
            args.rejected_models_output,
            error=exc,
            model=args.model,
            candidate=exc.candidate,
            diagnostics=exc.diagnostics,
        )
        raise
    quality = evaluate_emission_quality(
        payload,
        margin=args.emission_margin,
        min_negative_observations=args.min_negative_observations,
    )
    _write_json_atomic(payload, args.output)
    if args.compat_output:
        _write_json_atomic(payload, args.compat_output)
    Path(args.prompt_output).write_text(instructions + "\n", encoding="utf-8")
    Path(args.input_text_output).write_text(corpus_text + "\n", encoding="utf-8")
    _write_json_atomic(quality, args.quality_report_output)
    moves = Counter(
        move
        for row in records for turn in row["dialog"]
        for move in turn.get("annotated_teacher_moves", [])
    )
    manifest = {
        "input": args.input,
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "output": args.output,
        "output_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "analysis_model": args.model,
        "mock": args.mock,
        "conversations": len(records),
        "teacher_moves": dict(sorted(moves.items())),
        "prompt_sha256": _sha256_text(instructions),
        "analysis_text_sha256": _sha256_text(corpus_text),
        "max_input_chars": args.max_input_chars,
        "max_output_tokens": args.max_output_tokens,
        "emission_quality": quality,
    }
    _write_json_atomic(manifest, args.manifest)
    report(f"MathDial遷移ベイズモデルを書き出しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
