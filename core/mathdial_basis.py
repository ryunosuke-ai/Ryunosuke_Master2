"""MathDial向けontology、特徴抽出、ベイズモデルの共通処理。"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


MODEL_TYPE = "mathdial_basis_dialogue_v1"
REQUIRED_EXTRACTION_KEYS = (
    "student_state_before",
    "tutor_strategy",
    "student_state_after",
    "conversation_stage",
    "style_features",
    "confidence",
    "short_reason",
)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """YAML設定をobjectとして読み込む。"""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAMLのrootはobjectである必要があります: {path}")
    return payload


def file_sha256(path: Path | str) -> str:
    """ファイル内容のSHA-256を返す。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    """JSON互換値の安定hashを返す。"""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_ontology(ontology: dict[str, Any]) -> dict[str, Any]:
    """MathDial ontologyの必須集合と対応表を検証する。"""
    for key in ("name", "version"):
        if not str(ontology.get(key, "")).strip():
            raise ValueError(f"ontologyの`{key}`が空です。")
    for key in ("student_states", "tutor_strategies", "conversation_stages", "style_features"):
        values = ontology.get(key)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"ontologyの`{key}`は重複のない非空配列である必要があります。")
    mapping = ontology.get("teacher_move_mapping")
    if not isinstance(mapping, dict) or set(mapping) != {"probing", "focus", "telling", "generic"}:
        raise ValueError("teacher_move_mappingはMathDialの4カテゴリを完全に含む必要があります。")
    known = set(ontology["tutor_strategies"])
    mapped = [strategy for values in mapping.values() for strategy in values]
    if set(mapped) != known or len(mapped) != len(set(mapped)):
        raise ValueError("各tutor strategyはTeacher move上位カテゴリへ一意に対応する必要があります。")
    weights = ontology.get("score_weights", {})
    if set(weights) != {"state", "strategy", "transition", "style"}:
        raise ValueError("score_weightsのキーが不正です。")
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("score_weightsの合計は1である必要があります。")
    return ontology


def validate_extraction(payload: dict[str, Any], ontology: dict[str, Any]) -> dict[str, Any]:
    """LLM特徴抽出JSONを固定schemaへ正規化する。"""
    missing = [key for key in REQUIRED_EXTRACTION_KEYS if key not in payload]
    if missing:
        raise ValueError(f"特徴抽出JSONに不足があります: {missing}")
    label_specs = {
        "student_state_before": "student_states",
        "student_state_after": "student_states",
        "tutor_strategy": "tutor_strategies",
        "conversation_stage": "conversation_stages",
    }
    normalized: dict[str, Any] = {}
    for key, ontology_key in label_specs.items():
        value = str(payload[key]).strip()
        if value not in ontology[ontology_key]:
            raise ValueError(f"`{key}`に未知ラベルがあります: {value}")
        normalized[key] = value
    features = payload["style_features"]
    if not isinstance(features, list):
        raise ValueError("`style_features`は配列である必要があります。")
    unknown = sorted(set(map(str, features)) - set(ontology["style_features"]))
    if unknown:
        raise ValueError(f"style_featuresに未知ラベルがあります: {unknown}")
    normalized["style_features"] = list(dict.fromkeys(map(str, features)))
    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("`confidence`は0〜1の数値である必要があります。")
    normalized["confidence"] = float(confidence)
    reason = str(payload["short_reason"]).strip()
    if not reason:
        raise ValueError("`short_reason`が空です。")
    normalized["short_reason"] = reason
    return normalized


def build_extraction_instructions(ontology: dict[str, Any]) -> str:
    """Teacher moveを含まないMathDial特徴抽出指示を作る。"""
    schema = {
        "student_state_before": ontology["student_states"],
        "tutor_strategy": ontology["tutor_strategies"],
        "student_state_after": ontology["student_states"],
        "conversation_stage": ontology["conversation_stages"],
        "style_features": ontology["style_features"],
        "confidence": "0.0-1.0",
        "short_reason": "one short sentence",
    }
    return (
        "You analyze one turn of a one-to-one tutoring dialogue. The target is the pedagogical process, "
        "not mathematical topic similarity. Infer the learner state before the tutor response, the tutor's "
        "single primary strategy, the learner state after it, and the dialogue stage. Do not reward a response "
        "merely for giving a correct final answer. Distinguish diagnosis, focused guidance, graduated hints, "
        "explanation, and premature telling. If no following learner turn is observed, use unobserved for the "
        "after state. Return JSON only and use exactly one listed label for each scalar field.\n\n"
        f"Allowed schema and labels:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def build_validation_instructions(ontology: dict[str, Any]) -> str:
    """抽出とは独立したvalidation指示を作る。"""
    return (
        "Independently validate a structured tutoring-turn analysis against the transcript. Check that every "
        "label is supported by observable text, that the primary strategy is the best available label, and that "
        "an after-state is not invented without a following learner turn. Return JSON only with keys valid "
        "(boolean), corrected_extraction (the complete extraction schema), confidence (0-1), and short_reason.\n"
        f"Ontology: {json.dumps(ontology, ensure_ascii=False, sort_keys=True)}"
    )


def format_extraction_input(sample: dict[str, Any], conversation: dict[str, Any] | None = None) -> str:
    """特徴抽出入力を構造化する。Teacher move metadataは含めない。"""
    context = {}
    if conversation:
        metadata = conversation.get("metadata", {})
        context = {
            "problem": metadata.get("question"),
            "reference_answer": metadata.get("ground_truth"),
        }
    value = {
        "sample_id": sample["sample_id"],
        "task_context": context,
        "history": sample["history"],
        "tutor_response": sample["response"],
        "next_student_turn": sample.get("next_user_turn"),
    }
    return json.dumps(value, ensure_ascii=False, indent=2)


def _normalize(counter: dict[str, float], labels: Iterable[str], alpha: float) -> dict[str, float]:
    labels = list(labels)
    total = sum(float(counter.get(label, 0.0)) for label in labels) + alpha * len(labels)
    return {label: (float(counter.get(label, 0.0)) + alpha) / total for label in labels}


def _conditional(
    counts: dict[str, Counter[str]], rows: Iterable[str], columns: Iterable[str], alpha: float
) -> dict[str, dict[str, float]]:
    return {row: _normalize(counts.get(row, Counter()), columns, alpha) for row in rows}


def build_basis_model(extractions: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    """検証済みtrain抽出からMathDialベイズ対話モデルを構築する。"""
    validate_ontology(ontology)
    threshold = float(ontology.get("confidence_threshold", 0.0))
    usable = [row for row in extractions if row.get("split") == "train" and row.get("validation_status", "valid") == "valid" and float(row.get("confidence", 0.0)) >= threshold]
    if not usable:
        raise ValueError("ベイズモデル構築に利用できるtrain抽出がありません。")
    states = ontology["student_states"]
    strategies = ontology["tutor_strategies"]
    stages = ontology["conversation_stages"]
    styles = ontology["style_features"]
    alpha = float(ontology.get("smoothing_alpha", 0.5))
    state_counts: Counter[str] = Counter()
    strategy_by_state: dict[str, Counter[str]] = defaultdict(Counter)
    strategy_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    next_by_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    next_by_state_strategy: dict[str, Counter[str]] = defaultdict(Counter)
    style_counts: Counter[str] = Counter()
    for row in usable:
        weight = float(row["confidence"])
        before, strategy, after, stage = (row["student_state_before"], row["tutor_strategy"], row["student_state_after"], row["conversation_stage"])
        state_counts[before] += weight
        strategy_by_state[before][strategy] += weight
        strategy_by_stage[stage][strategy] += weight
        if after != "unobserved":
            next_by_strategy[strategy][after] += weight
            next_by_state_strategy[f"{before}|{strategy}"][after] += weight
        for feature in row["style_features"]:
            style_counts[feature] += weight
    model = {
        "name": "mathdial_tutoring_basis",
        "model_type": MODEL_TYPE,
        "ontology_version": ontology["version"],
        "states": states,
        "strategies": strategies,
        "stages": stages,
        "style_features": styles,
        "smoothing_alpha": alpha,
        "confidence_threshold": threshold,
        "score_weights": ontology["score_weights"],
        "state_prior": _normalize(state_counts, states, alpha),
        "strategy_given_state": _conditional(strategy_by_state, states, strategies, alpha),
        "strategy_given_stage": _conditional(strategy_by_stage, stages, strategies, alpha),
        "next_state_given_strategy": _conditional(next_by_strategy, strategies, states, alpha),
        "next_state_given_state_strategy": _conditional(next_by_state_strategy, (f"{s}|{t}" for s in states for t in strategies), states, alpha),
        "style_prior": _normalize(style_counts, styles, alpha),
        "training_records": len(usable),
        "frequencies": {
            "states": dict(state_counts),
            "strategies": dict(sum(strategy_by_state.values(), Counter())),
            "stages": dict(Counter(row["conversation_stage"] for row in usable)),
            "styles": dict(style_counts),
        },
    }
    calibration = {key: [] for key in ("state", "strategy", "transition", "style")}
    for row in usable:
        raw = raw_component_scores(row, model)
        for key, value in raw.items():
            if value is not None:
                calibration[key].append(value)
    model["calibration"] = {key: sorted(values) for key, values in calibration.items()}
    model["model_sha256"] = canonical_hash({key: value for key, value in model.items() if key != "model_sha256"})
    return model


def _pedagogical_state(row: dict[str, Any]) -> str:
    """細粒度ラベルをESConv互換の会話進行状態へ写像する。"""
    strategy = row["tutor_strategy"]
    before = row["student_state_before"]
    stage = row["conversation_stage"]
    if strategy == "direct_telling" and before not in {"corrected_understanding", "solution_reached"}:
        return "premature_telling"
    if strategy == "conversation_management" and stage not in {"problem_understanding", "closure"}:
        return "generic_ungrounded"
    if stage in {"problem_understanding", "misconception_diagnosis"}:
        return "diagnosing"
    if stage in {"guided_reasoning", "focused_scaffolding"}:
        return "scaffolding"
    if stage == "explicit_explanation":
        return "explaining"
    return "verifying"


def build_transition_compat_model(extractions: list[dict[str, Any]], ontology: dict[str, Any]) -> dict[str, Any]:
    """既存ESConv scorerで利用できるtransition_bayes_network viewを作る。"""
    threshold = float(ontology.get("confidence_threshold", 0.0))
    usable = [row for row in extractions if row.get("split") == "train" and row.get("validation_status", "valid") == "valid" and float(row.get("confidence", 0.0)) >= threshold]
    if not usable:
        raise ValueError("ESConv互換モデルを構築できるtrain抽出がありません。")
    states = ("diagnosing", "scaffolding", "explaining", "verifying", "premature_telling", "generic_ungrounded")
    observations = tuple(ontology["tutor_strategies"])
    alpha = float(ontology.get("smoothing_alpha", 0.5))
    initial: Counter[str] = Counter()
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    emissions: dict[str, Counter[str]] = defaultdict(Counter)
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_conversation[str(row["conversation_id"])].append(row)
        state = _pedagogical_state(row)
        emissions[state][row["tutor_strategy"]] += float(row["confidence"])
    for rows in by_conversation.values():
        rows.sort(key=lambda item: int(item.get("assistant_turn_index", 0)))
        state_sequence = [_pedagogical_state(row) for row in rows]
        initial[state_sequence[0]] += 1.0
        for left, right in zip(state_sequence, state_sequence[1:]):
            transitions[left][right] += 1.0
    descriptions = {
        "diagnosing": "学習者の試行、誤り、理解状態を確認している段階。",
        "scaffolding": "質問や段階的ヒントで学習者自身の推論を進める望ましい段階。",
        "explaining": "診断後に必要な概念や手順を明示的に説明する段階。",
        "verifying": "自己修正、理解、解決を確認して会話をまとめる段階。",
        "premature_telling": "十分な診断や足場かけなしに答えを直接明かす望ましくない段階。",
        "generic_ungrounded": "学習者の状態や直前の内容に根差さない一般的応答の段階。",
    }
    observation_descriptions = {strategy: f"MathDial ontologyのtutor strategy: {strategy}" for strategy in observations}
    return {
        "name": "mathdial_tutoring_transition_compat_v1",
        "model_type": "transition_bayes_network",
        "states": list(states),
        "positive_states": ["diagnosing", "scaffolding", "explaining", "verifying"],
        "negative_states": ["premature_telling", "generic_ungrounded"],
        "observations": list(observations),
        "initial_state_prior": _normalize(initial, states, alpha),
        "transition_likelihoods": _conditional(transitions, states, states, alpha),
        "emission_likelihoods": _conditional(emissions, states, observations, alpha),
        "state_descriptions": descriptions,
        "observation_descriptions": observation_descriptions,
        "dataset_hypothesis": "学習者の誤りや混乱を診断し、早すぎる解答提示を避け、質問と段階的ヒントで自己修正を促す個別指導スタイル。",
        "metadata": {
            "ontology_version": ontology["version"],
            "confidence_threshold": threshold,
            "smoothing_alpha": alpha,
            "training_records": len(usable),
            "fine_grained_model_type": MODEL_TYPE,
        },
    }


def raw_component_scores(row: dict[str, Any], model: dict[str, Any]) -> dict[str, float | None]:
    """抽出ラベルの対数尤度成分を返す。"""
    before, strategy, after, stage = (row["student_state_before"], row["tutor_strategy"], row["student_state_after"], row["conversation_stage"])
    state = math.log(model["state_prior"][before])
    strategy_score = 0.5 * (math.log(model["strategy_given_state"][before][strategy]) + math.log(model["strategy_given_stage"][stage][strategy]))
    transition = None
    if after != "unobserved":
        transition = 0.5 * (math.log(model["next_state_given_strategy"][strategy][after]) + math.log(model["next_state_given_state_strategy"][f"{before}|{strategy}"][after]))
    features = row.get("style_features", [])
    style = sum(math.log(model["style_prior"][feature]) for feature in features) / len(features) if features else math.log(1e-12)
    return {"state": state, "strategy": strategy_score, "transition": transition, "style": style}


def _percentile(sorted_values: list[float], value: float | None) -> float | None:
    if value is None:
        return None
    if not sorted_values:
        return 0.5
    return bisect.bisect_right(sorted_values, value) / len(sorted_values)


def score_extraction(row: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """抽出ラベルをBASiS component scoreへ変換する。"""
    raw = raw_component_scores(row, model)
    scores = {key: _percentile(model["calibration"].get(key, []), value) for key, value in raw.items()}
    weights = dict(model["score_weights"])
    active = {key: weight for key, weight in weights.items() if scores[key] is not None}
    total = sum(active.values())
    basis = sum((active[key] / total) * float(scores[key]) for key in active)
    return {"basis_score": basis, "state_score": scores["state"], "strategy_score": scores["strategy"], "transition_score": scores["transition"], "style_score": scores["style"], "raw_component_scores": raw}
