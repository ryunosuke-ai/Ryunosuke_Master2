"""状態遷移を持つ生成ベイズモデルの検証と更新処理。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EPSILON = 1e-12
MODEL_TYPE = "transition_bayes_network"


@dataclass(frozen=True)
class TransitionObservationScore:
    """1ターンの観測評価。"""

    observation: str
    score: float
    reason: str = ""


@dataclass(frozen=True)
class TransitionBayesModel:
    """LLMが生成した状態遷移ベイズモデル仕様。"""

    name: str
    model_type: str
    states: tuple[str, ...]
    positive_states: tuple[str, ...]
    negative_states: tuple[str, ...]
    observations: tuple[str, ...]
    initial_state_prior: dict[str, float]
    transition_likelihoods: dict[str, dict[str, float]]
    emission_likelihoods: dict[str, dict[str, float]]
    state_descriptions: dict[str, str]
    observation_descriptions: dict[str, str]
    dataset_hypothesis: str


def _require_mapping(payload: Any, name: str) -> dict[str, Any]:
    """値がdictであることを検証する。"""
    if not isinstance(payload, dict):
        raise ValueError(f"`{name}` はオブジェクトである必要があります。")
    return payload


def _require_nonempty_string(payload: dict[str, Any], key: str) -> str:
    """必須文字列を取り出す。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` は空でない文字列である必要があります。")
    return value.strip()


def _require_probability(value: Any, key: str) -> float:
    """確率値として妥当かを検証する。"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` は数値である必要があります。")
    probability = float(value)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"`{key}` は 0.0 より大きく 1.0 より小さい必要があります。")
    return probability


def _require_label_list(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    """ラベル配列を検証してtupleへ変換する。"""
    raw_items = payload.get(key)
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"`{key}` は空でない配列である必要があります。")
    labels = tuple(str(item).strip() for item in raw_items if str(item).strip())
    if not labels:
        raise ValueError(f"`{key}` は空でない配列である必要があります。")
    if len(labels) != len(set(labels)):
        raise ValueError(f"`{key}` に重複があります。")
    return labels


def _require_subset(values: tuple[str, ...], candidates: tuple[str, ...], key: str) -> None:
    """ラベル配列が候補集合に含まれることを検証する。"""
    candidate_set = set(candidates)
    unknown = [value for value in values if value not in candidate_set]
    if unknown:
        raise ValueError(f"`{key}` に未知のラベルがあります: {unknown}")


def _normalize_probability_row(
    row: Any,
    *,
    row_name: str,
    columns: tuple[str, ...],
) -> dict[str, float]:
    """確率行を検証して正規化する。"""
    mapping = _require_mapping(row, row_name)
    normalized: dict[str, float] = {}
    for column in columns:
        normalized[column] = _require_probability(mapping.get(column), f"{row_name}.{column}")
    total = sum(normalized.values())
    if abs(total - 1.0) > 0.05:
        raise ValueError(f"`{row_name}` の合計は1.0付近である必要があります。現在値: {total:.4f}")
    return {key: value / total for key, value in normalized.items()}


def load_transition_bayes_model(path: Path | str) -> TransitionBayesModel:
    """JSONファイルから状態遷移ベイズモデルを読み込む。"""
    model_path = Path(path)
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"状態遷移ベイズモデルJSONが見つかりません: {model_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"状態遷移ベイズモデルJSONを解析できません: {exc}") from exc
    return parse_transition_bayes_model(payload)


def parse_transition_bayes_model(payload: dict[str, Any]) -> TransitionBayesModel:
    """dictから状態遷移ベイズモデルを構築する。"""
    root = _require_mapping(payload, "root")
    name = _require_nonempty_string(root, "name")
    model_type = _require_nonempty_string(root, "model_type")
    if model_type != MODEL_TYPE:
        raise ValueError(f"`model_type` は {MODEL_TYPE!r} である必要があります。")

    states = _require_label_list(root, "states")
    positive_states = _require_label_list(root, "positive_states")
    negative_states = _require_label_list(root, "negative_states")
    observations = _require_label_list(root, "observations")
    _require_subset(positive_states, states, "positive_states")
    _require_subset(negative_states, states, "negative_states")
    if set(positive_states) & set(negative_states):
        raise ValueError("`positive_states` と `negative_states` は重複しない必要があります。")

    initial_state_prior = _normalize_probability_row(
        root.get("initial_state_prior"),
        row_name="initial_state_prior",
        columns=states,
    )

    transition_raw = _require_mapping(root.get("transition_likelihoods"), "transition_likelihoods")
    transition_likelihoods = {
        state: _normalize_probability_row(
            transition_raw.get(state),
            row_name=f"transition_likelihoods.{state}",
            columns=states,
        )
        for state in states
    }

    emission_raw = _require_mapping(root.get("emission_likelihoods"), "emission_likelihoods")
    emission_likelihoods = {
        state: _normalize_probability_row(
            emission_raw.get(state),
            row_name=f"emission_likelihoods.{state}",
            columns=observations,
        )
        for state in states
    }

    state_descriptions = {
        str(key): str(value)
        for key, value in _require_mapping(root.get("state_descriptions"), "state_descriptions").items()
    }
    observation_descriptions = {
        str(key): str(value)
        for key, value in _require_mapping(root.get("observation_descriptions"), "observation_descriptions").items()
    }
    dataset_hypothesis = _require_nonempty_string(root, "dataset_hypothesis")

    missing_states = [state for state in states if state not in state_descriptions]
    if missing_states:
        raise ValueError(f"`state_descriptions` に不足があります: {missing_states}")
    missing_observations = [item for item in observations if item not in observation_descriptions]
    if missing_observations:
        raise ValueError(f"`observation_descriptions` に不足があります: {missing_observations}")

    return TransitionBayesModel(
        name=name,
        model_type=model_type,
        states=states,
        positive_states=positive_states,
        negative_states=negative_states,
        observations=observations,
        initial_state_prior=initial_state_prior,
        transition_likelihoods=transition_likelihoods,
        emission_likelihoods=emission_likelihoods,
        state_descriptions=state_descriptions,
        observation_descriptions=observation_descriptions,
        dataset_hypothesis=dataset_hypothesis,
    )


def predict_next_state_distribution(
    model: TransitionBayesModel,
    prior_distribution: dict[str, float] | None = None,
) -> dict[str, float]:
    """遷移確率に基づき、次ターン前の状態分布を予測する。"""
    prior = model.initial_state_prior if prior_distribution is None else prior_distribution
    predicted = {state: 0.0 for state in model.states}
    for previous_state in model.states:
        previous_probability = float(prior.get(previous_state, 0.0))
        for next_state in model.states:
            predicted[next_state] += previous_probability * model.transition_likelihoods[previous_state][next_state]
    total = sum(predicted.values())
    if total <= DEFAULT_EPSILON:
        return dict(model.initial_state_prior)
    return {state: max(DEFAULT_EPSILON, value / total) for state, value in predicted.items()}


def update_state_distribution(
    model: TransitionBayesModel,
    prior_distribution: dict[str, float] | None,
    observation: str,
) -> dict[str, float]:
    """状態遷移と観測ラベルに基づき、状態分布を更新する。"""
    if observation not in model.observations:
        raise ValueError(f"未知の観測ラベルです: {observation}")
    predicted = predict_next_state_distribution(model, prior_distribution)
    weighted = {
        state: predicted[state] * model.emission_likelihoods[state][observation]
        for state in model.states
    }
    evidence = sum(weighted.values())
    if evidence <= DEFAULT_EPSILON:
        return predicted
    return {state: max(DEFAULT_EPSILON, value / evidence) for state, value in weighted.items()}


def positive_posterior(model: TransitionBayesModel, state_distribution: dict[str, float]) -> float:
    """望ましい状態群の合計確率を返す。"""
    posterior = sum(float(state_distribution.get(state, 0.0)) for state in model.positive_states)
    return max(0.001, min(0.999, posterior))


def score_transition_observation(
    model: TransitionBayesModel,
    score: TransitionObservationScore,
    prior_distribution: dict[str, float] | None = None,
) -> dict[str, Any]:
    """観測評価を状態分布更新結果付きのdictへ変換する。"""
    start_distribution = model.initial_state_prior if prior_distribution is None else prior_distribution
    predicted_distribution = predict_next_state_distribution(model, start_distribution)
    state_posteriors = update_state_distribution(model, start_distribution, score.observation)
    prior_positive = positive_posterior(model, predicted_distribution)
    posterior = positive_posterior(model, state_posteriors)
    most_likely_state = max(state_posteriors, key=state_posteriors.get)
    return {
        "observation": score.observation,
        "observation_score": score.score,
        "reason": score.reason,
        "prior": prior_positive,
        "posterior": posterior,
        "delta": posterior - prior_positive,
        "state_posteriors": state_posteriors,
        "most_likely_state": most_likely_state,
    }
