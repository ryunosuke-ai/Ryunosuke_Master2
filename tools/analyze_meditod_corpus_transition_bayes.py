"""MediTOD完全診療と公式annotation集計から遷移ベイズモデルを直接生成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from core.transition_bayes_model import parse_transition_bayes_model
from tools.analyze_mathdial_corpus_transition_bayes import evaluate_emission_quality
from tools.analyze_small_corpus import (
    OpenAIResponsesGenerator,
    TextGenerator,
    extract_json_object,
    load_env_file,
    resolve_analysis_model,
)
from tools.analyze_small_corpus_transition_bayes import build_json_repair_instructions


DEFAULT_MAX_OUTPUT_TOKENS = 24_000
# 公開raw版の層化24診療は、公式annotationを含めると約68万文字になる。
# 診療境界を壊さず全標本を渡せるよう、実測値に十分な余裕を持たせる。
DEFAULT_MAX_INPUT_CHARS = 800_000
REQUIRED_FUNCTIONS = {
    "complaint_elicitation": ("complaint", "open_elicitation"),
    "symptom_attributes": ("symptom_attribute", "onset", "severity", "progression"),
    "associated_red_flags": ("associated", "red_flag"),
    "background_history": ("history", "medication", "test", "exposure", "habit"),
    "summary_transition": ("summary", "transition"),
    "premature_assessment": ("premature", "diagnosis", "advice"),
    "redundancy_misalignment": ("redundant", "repetition", "misaligned", "context"),
}


class MediTODModelQualityError(ValueError):
    """生成候補がMediTOD品質gateを満たさない場合の例外。"""

    def __init__(self, message: str, *, candidate: dict[str, Any], diagnostics: dict[str, Any]):
        super().__init__(message)
        self.candidate = candidate
        self.diagnostics = diagnostics


def read_analysis_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError("MediTOD分析標本が空です。")
    if any(row.get("source_split") != "train" for row in rows):
        raise ValueError("MediTOD分析標本へtrain以外が混入しています。")
    ids = [str(row.get("conversation_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("MediTOD分析標本のconversation_idが空または重複しています。")
    return rows


def build_meditod_corpus_text(
    records: list[dict[str, Any]],
    aggregates: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    sections = [
        "# deterministic_train_annotation_aggregates",
        json.dumps(aggregates, ensure_ascii=False, indent=2),
        "# stratified_complete_consultations",
    ]
    for record in records:
        sections.append(f"\n## conversation_id={record['conversation_id']}")
        for turn in record["dialog"]:
            annotations = json.dumps(
                turn.get("annotation_variants", []), ensure_ascii=False, separators=(",", ":")
            )
            sections.append(
                f"{int(turn['turn_index'])}. {turn['speaker']} "
                f"[official_annotations={annotations}]: {turn['text']}"
            )
    text = "\n".join(sections).strip()
    if len(text) > max_chars:
        raise ValueError(
            "MediTOD分析入力が上限を超えています。完全診療を切らないため、"
            f"--max-input-charsを増やしてください: {len(text)}/{max_chars}"
        )
    return text


def build_meditod_analysis_instructions() -> str:
    return """あなたは医療対話コーパス分析、病歴聴取、動的ベイズモデル設計の専門家です。

以下のMediTOD小規模コーパスを分析し、このコーパスが表現する体系的な病歴聴取の進め方を状態遷移ベイズモデルとして作成してください。目的は医学的診断知識そのものではなく、情報不足を認識し、症状属性を確認し、関連症状から既往歴・服薬・検査・生活背景へ順序立てて移る医療者側の会話スタイルをWildChatから選別することです。

入力には、train全体から決定論的に集計したintent/slot/attribute頻度、会話十分位別slot分布、doctor slot遷移、doctor actionから次patient informationへの遷移と、層化した完全診療が含まれます。official_annotationsは高価値な外部annotationとして強く参照してください。ただし、ラベル名をそのままstate名へコピーせず、会話本文、段階、次の患者情報と照合してください。

必ず区別する機能:
- 主訴を開放的に聴取する。
- 発症時期、期間、経過、重症度、特徴を確認する。
- 関連症状とred flagを確認する。
- 既往歴、家族歴、服薬、検査、習慣、曝露、旅行、生活背景を確認する。
- 既知情報を要約し、適切な次段階へ移る。
- 情報不足のまま診断や対応方針を断定する早すぎるassessment/adviceを識別する。
- すでに得た情報の不要な反復質問と、直前文脈に合わない質問・応答を識別する。

モデル設計方針:
1. statesは病歴聴取の進行局面を表す4〜7個とする。
2. positive_statesは不足情報の認識、適切な質問、情報統合、段階移行を表す。
3. negative_statesは早すぎる診断・助言、重複質問、文脈不一致を表す。
4. observationsはprompt/responseから後段LLMが直接分類できる医療者応答機能を6〜10個作る。
5. state名とobservation名を機能的に重複させない。
6. 各正負stateに反対群よりemissionが0.10以上高い識別observationを持たせる。
7. negative優勢observationは最低2種類とし、premature assessmentとredundancy/misalignmentを統合しない。
8. 病歴聴取の質問と、情報が揃った後の適切な要約・説明を両方正当に扱う。

出力制約:
- JSON objectのみ。Markdownや前後説明は禁止。
- model_typeはtransition_bayes_network。
- 必須キーはname, model_type, states, positive_states, negative_states, observations, initial_state_prior, transition_likelihoods, emission_likelihoods, state_descriptions, observation_descriptions, dataset_hypothesis。
- statesは4〜7個、observationsは6〜10個。
- ラベルは英小文字・数字・アンダースコアだけ。
- positive_statesとnegative_statesは重複せず、全statesを被覆する。
- priorと各確率行は全ラベルを含み、各値は0より大きく1より小さく、行合計を1.0にする。
- descriptionsとdataset_hypothesisは、医学知識の正しさではなく対話機能を分類できる具体的な日本語で書く。

以下が分析対象です。""".strip()


def mock_model() -> dict[str, Any]:
    states = [
        "complaint_exploration",
        "symptom_characterization",
        "background_history_collection",
        "integrated_stage_transition",
        "premature_assessment",
        "repetitive_misalignment",
    ]
    observations = [
        "open_complaint_elicitation",
        "symptom_attribute_question",
        "associated_or_red_flag_question",
        "background_information_question",
        "summary_or_stage_transition",
        "premature_diagnosis_or_advice",
        "redundant_question",
        "context_misaligned_response",
    ]
    return {
        "name": "meditod_history_taking_transition_model",
        "model_type": "transition_bayes_network",
        "states": states,
        "positive_states": states[:4],
        "negative_states": states[4:],
        "observations": observations,
        "initial_state_prior": dict(zip(states, [0.30, 0.28, 0.18, 0.10, 0.07, 0.07])),
        "transition_likelihoods": {
            states[0]: dict(zip(states, [0.30, 0.45, 0.10, 0.06, 0.04, 0.05])),
            states[1]: dict(zip(states, [0.08, 0.45, 0.28, 0.10, 0.04, 0.05])),
            states[2]: dict(zip(states, [0.05, 0.12, 0.48, 0.25, 0.05, 0.05])),
            states[3]: dict(zip(states, [0.05, 0.08, 0.17, 0.55, 0.08, 0.07])),
            states[4]: dict(zip(states, [0.08, 0.08, 0.08, 0.08, 0.60, 0.08])),
            states[5]: dict(zip(states, [0.10, 0.10, 0.10, 0.08, 0.07, 0.55])),
        },
        "emission_likelihoods": {
            states[0]: dict(zip(observations, [0.50, 0.12, 0.10, 0.08, 0.08, 0.04, 0.04, 0.04])),
            states[1]: dict(zip(observations, [0.08, 0.40, 0.28, 0.08, 0.06, 0.04, 0.03, 0.03])),
            states[2]: dict(zip(observations, [0.06, 0.12, 0.14, 0.44, 0.12, 0.04, 0.04, 0.04])),
            states[3]: dict(zip(observations, [0.06, 0.10, 0.10, 0.14, 0.44, 0.06, 0.05, 0.05])),
            states[4]: dict(zip(observations, [0.05, 0.06, 0.07, 0.07, 0.10, 0.52, 0.06, 0.07])),
            states[5]: dict(zip(observations, [0.05, 0.06, 0.06, 0.07, 0.07, 0.07, 0.36, 0.26])),
        },
        "state_descriptions": {
            states[0]: "患者の主訴と困り事を開放的に聴き、調査対象を定める局面。",
            states[1]: "症状の発症時期、期間、経過、重症度、特徴、関連症状やred flagを具体化する局面。",
            states[2]: "既往歴、家族歴、服薬、検査、習慣、曝露、旅行、生活背景を収集する局面。",
            states[3]: "既知情報を要約し、情報充足を確認して次の病歴聴取段階へ移る局面。",
            states[4]: "必要な病歴情報がないまま診断や助言を断定する局面。",
            states[5]: "取得済み情報を重複質問するか直前の患者情報と対応せず停滞する局面。",
        },
        "observation_descriptions": {
            observations[0]: "患者自身の言葉で主訴を話せる開放的な質問。",
            observations[1]: "症状のonset、期間、progression、severity、characteristicsを確認する質問。",
            observations[2]: "関連症状または緊急性に関わるred flagを確認する質問。",
            observations[3]: "既往歴、家族歴、medication、medical test、habit、exposure、travel、生活背景を確認する質問。",
            observations[4]: "得られた情報をsummaryし、情報不足に沿って次のstageへtransitionする応答。",
            observations[5]: "情報不足のままpremature diagnosisまたは具体的adviceを断定する応答。",
            observations[6]: "すでに回答済みの情報を不要にrepetitionするredundant question。",
            observations[7]: "直前の患者情報と対応しないcontext-misaligned response。",
        },
        "dataset_hypothesis": "不足情報を認識し、症状属性から背景歴へ段階的に移る体系的な病歴聴取対話。",
    }


def evaluate_functional_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    model = parse_transition_bayes_model(payload)
    searchable = {
        observation: f"{observation} {model.observation_descriptions[observation]}".lower()
        for observation in model.observations
    }
    rows = {}
    for function, keywords in REQUIRED_FUNCTIONS.items():
        matched = [
            observation
            for observation, text in searchable.items()
            if any(keyword in text for keyword in keywords)
        ]
        rows[function] = {"matched_observations": matched, "passed": bool(matched)}
    return {"passed": all(row["passed"] for row in rows.values()), "functions": rows}


def evaluate_model_quality(payload: dict[str, Any], *, emission_margin: float, min_negative: int) -> dict[str, Any]:
    emission = evaluate_emission_quality(
        payload, margin=emission_margin, min_negative_observations=min_negative
    )
    coverage = evaluate_functional_coverage(payload)
    model = parse_transition_bayes_model(payload)
    shape = {
        "states": len(model.states),
        "observations": len(model.observations),
        "positive_negative_cover_all_states": (
            set(model.positive_states) | set(model.negative_states) == set(model.states)
        ),
        "passed": (
            4 <= len(model.states) <= 7
            and 6 <= len(model.observations) <= 10
            and set(model.positive_states) | set(model.negative_states) == set(model.states)
        ),
    }
    return {
        "passed": emission["passed"] and coverage["passed"] and shape["passed"],
        "emission_quality": emission,
        "functional_coverage": coverage,
        "shape": shape,
    }


def generate_model(
    records: list[dict[str, Any]],
    aggregates: dict[str, Any],
    *,
    generator: TextGenerator | None,
    model: str,
    max_output_tokens: int,
    max_input_chars: int,
    mock: bool,
    emission_margin: float,
    min_negative: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    instructions = build_meditod_analysis_instructions()
    corpus = build_meditod_corpus_text(records, aggregates, max_chars=max_input_chars)
    if mock:
        payload = mock_model()
    else:
        assert generator is not None
        if progress:
            progress(f"{model}へMediTOD遷移ベイズモデル生成を依頼しています。")
        raw = generator.generate(
            instructions=instructions,
            input_text=corpus,
            model=model,
            max_output_tokens=max_output_tokens,
            response_text_format={"type": "json_object"},
        )
        try:
            payload = extract_json_object(raw)
        except ValueError:
            repaired = generator.generate(
                instructions=build_json_repair_instructions(),
                input_text=raw,
                model=model,
                max_output_tokens=max_output_tokens,
                response_text_format={"type": "json_object"},
            )
            payload = extract_json_object(repaired)
    try:
        quality = evaluate_model_quality(
            payload, emission_margin=emission_margin, min_negative=min_negative
        )
        if not quality["passed"]:
            raise ValueError("MediTODベイズモデル品質gateに不合格です。")
    except ValueError as exc:
        raise MediTODModelQualityError(
            str(exc), candidate=payload, diagnostics=locals().get("quality", {"passed": False})
        ) from exc
    return payload, instructions, corpus, quality


def _write_json(payload: dict[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="MediTODから遷移ベイズモデルを生成")
    parser.add_argument("--input", required=True)
    parser.add_argument("--aggregates", required=True)
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
    parser.add_argument("--emission-margin", type=float, default=0.10)
    parser.add_argument("--min-negative-observations", type=int, default=2)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    records = read_analysis_jsonl(args.input)
    aggregates = json.loads(Path(args.aggregates).read_text(encoding="utf-8"))
    started = time.monotonic()

    def progress(message: str) -> None:
        print(f"[{time.monotonic() - started:6.1f}s] {message}", flush=True)

    try:
        payload, instructions, corpus, quality = generate_model(
            records,
            aggregates,
            generator=None if args.mock else OpenAIResponsesGenerator(),
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            max_input_chars=args.max_input_chars,
            mock=args.mock,
            emission_margin=args.emission_margin,
            min_negative=args.min_negative_observations,
            progress=progress,
        )
    except MediTODModelQualityError as exc:
        path = Path(args.rejected_models_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"model": args.model, "error": str(exc), "diagnostics": exc.diagnostics, "candidate": exc.candidate}, ensure_ascii=False) + "\n")
        raise
    _write_json(payload, args.output)
    if args.compat_output:
        _write_json(payload, args.compat_output)
    Path(args.prompt_output).write_text(instructions + "\n", encoding="utf-8")
    Path(args.input_text_output).write_text(corpus + "\n", encoding="utf-8")
    _write_json(quality, args.quality_report_output)
    annotation_counts = Counter(
        str(annotation.get("intent", "")).lower()
        for row in records
        for turn in row["dialog"]
        for variant in turn.get("annotation_variants", [])
        for annotation in variant.get("annotations", [])
        if annotation.get("intent")
    )
    manifest = {
        "input": args.input,
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "aggregates_sha256": hashlib.sha256(Path(args.aggregates).read_bytes()).hexdigest(),
        "output": args.output,
        "output_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "analysis_model": args.model,
        "mock": args.mock,
        "conversations": len(records),
        "turns": sum(len(row["dialog"]) for row in records),
        "intent_counts": dict(annotation_counts),
        "prompt_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
        "analysis_text_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "quality": quality,
    }
    _write_json(manifest, args.manifest)
    progress(f"MediTOD遷移ベイズモデルを書き出しました: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
