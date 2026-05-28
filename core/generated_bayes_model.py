"""生成されたベイズモデルを読み込み、対話評価に使うための純関数群。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True)
class ObservationScore:
    """1ターンの観測評価。"""

    observation: str
    score: float
    reason: str = ""


@dataclass(frozen=True)
class BayesModel:
    """LLMが生成したベイズモデル仕様。"""

    name: str
    positive_state: str
    negative_state: str
    observations: tuple[str, ...]
    likelihoods: dict[str, dict[str, float]]
    prior: float
    strategy_descriptions: dict[str, str]


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


def _normalize_likelihood_row(row: Any, *, state: str, observations: tuple[str, ...]) -> dict[str, float]:
    """1状態分の尤度テーブルを検証して正規化する。"""
    mapping = _require_mapping(row, f"likelihoods.{state}")
    normalized: dict[str, float] = {}
    for observation in observations:
        normalized[observation] = _require_probability(
            mapping.get(observation),
            f"likelihoods.{state}.{observation}",
        )
    total = sum(normalized.values())
    if abs(total - 1.0) > 0.05:
        raise ValueError(f"`likelihoods.{state}` の合計は1.0付近である必要があります。現在値: {total:.4f}")
    return normalized


def load_bayes_model(path: Path | str) -> BayesModel:
    """JSONファイルからベイズモデルを読み込む。"""
    model_path = Path(path)
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ベイズモデルJSONが見つかりません: {model_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"ベイズモデルJSONを解析できません: {exc}") from exc
    return parse_bayes_model(payload)


def parse_bayes_model(payload: dict[str, Any]) -> BayesModel:
    """dictからベイズモデルを構築する。"""
    root = _require_mapping(payload, "root")
    name = _require_nonempty_string(root, "name")
    positive_state = _require_nonempty_string(root, "positive_state")
    negative_state = _require_nonempty_string(root, "negative_state")

    observations_raw = root.get("observations")
    if not isinstance(observations_raw, list) or not observations_raw:
        raise ValueError("`observations` は空でない配列である必要があります。")
    observations = tuple(str(item).strip() for item in observations_raw if str(item).strip())
    if len(observations) != len(set(observations)):
        raise ValueError("`observations` に重複があります。")

    likelihoods_raw = _require_mapping(root.get("likelihoods"), "likelihoods")
    likelihoods = {
        positive_state: _normalize_likelihood_row(
            likelihoods_raw.get(positive_state),
            state=positive_state,
            observations=observations,
        ),
        negative_state: _normalize_likelihood_row(
            likelihoods_raw.get(negative_state),
            state=negative_state,
            observations=observations,
        ),
    }
    prior = _require_probability(root.get("prior", 0.5), "prior")

    strategies_raw = root.get("strategy_descriptions", {})
    strategies = {
        str(key): str(value)
        for key, value in _require_mapping(strategies_raw, "strategy_descriptions").items()
    }

    return BayesModel(
        name=name,
        positive_state=positive_state,
        negative_state=negative_state,
        observations=observations,
        likelihoods=likelihoods,
        prior=prior,
        strategy_descriptions=strategies,
    )


def update_posterior(model: BayesModel, prior: float, observation: str) -> float:
    """観測ラベルに基づき、positive_stateの事後確率を更新する。"""
    if observation not in model.observations:
        raise ValueError(f"未知の観測ラベルです: {observation}")
    bounded_prior = max(0.001, min(0.999, float(prior)))
    positive_likelihood = model.likelihoods[model.positive_state][observation]
    negative_likelihood = model.likelihoods[model.negative_state][observation]
    evidence = positive_likelihood * bounded_prior + negative_likelihood * (1.0 - bounded_prior)
    if evidence <= DEFAULT_EPSILON:
        return bounded_prior
    posterior = positive_likelihood * bounded_prior / evidence
    return max(0.001, min(0.999, posterior))


def score_observation(model: BayesModel, score: ObservationScore, prior: float | None = None) -> dict[str, Any]:
    """観測評価をベイズ更新結果付きのdictへ変換する。"""
    start_prior = model.prior if prior is None else prior
    posterior = update_posterior(model, start_prior, score.observation)
    return {
        "observation": score.observation,
        "observation_score": score.score,
        "reason": score.reason,
        "prior": start_prior,
        "posterior": posterior,
        "delta": posterior - start_prior,
    }
