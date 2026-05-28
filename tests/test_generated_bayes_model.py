"""生成ベイズモデルの読み込みと更新処理のテスト。"""

import pytest

from core.generated_bayes_model import ObservationScore, parse_bayes_model, score_observation, update_posterior


def make_model_payload():
    """テスト用ベイズモデルを返す。"""
    return {
        "name": "supportive_dialogue",
        "positive_state": "target_style",
        "negative_state": "non_target_style",
        "observations": ["deepening", "generic", "blocking"],
        "likelihoods": {
            "target_style": {
                "deepening": 0.7,
                "generic": 0.2,
                "blocking": 0.1,
            },
            "non_target_style": {
                "deepening": 0.1,
                "generic": 0.3,
                "blocking": 0.6,
            },
        },
        "prior": 0.5,
        "strategy_descriptions": {
            "deepening": "相手の内容を拾って自然に深める",
        },
    }


def test_parse_bayes_model_accepts_valid_payload():
    model = parse_bayes_model(make_model_payload())

    assert model.name == "supportive_dialogue"
    assert model.observations == ("deepening", "generic", "blocking")
    assert model.prior == 0.5


def test_update_posterior_increases_for_positive_observation():
    model = parse_bayes_model(make_model_payload())

    posterior = update_posterior(model, 0.5, "deepening")

    assert posterior > 0.5


def test_update_posterior_rejects_unknown_observation():
    model = parse_bayes_model(make_model_payload())

    with pytest.raises(ValueError, match="未知の観測ラベル"):
        update_posterior(model, 0.5, "unknown")


def test_score_observation_returns_delta():
    model = parse_bayes_model(make_model_payload())

    result = score_observation(model, ObservationScore("blocking", 0.9, "会話を止めている"), prior=0.5)

    assert result["posterior"] < 0.5
    assert result["delta"] < 0.0
    assert result["reason"] == "会話を止めている"


def test_parse_bayes_model_rejects_bad_likelihood_sum():
    payload = make_model_payload()
    payload["likelihoods"]["target_style"]["deepening"] = 0.2

    with pytest.raises(ValueError, match="合計"):
        parse_bayes_model(payload)
