# Master2 研究計画

## 目的

小さい会話コーパスから、LLMで会話特徴・観測ラベル・会話戦略を分析し、そのコーパスらしさを表すベイズモデルを自動生成する。生成したベイズモデルで大きな対話データを評価し、高スコア応答と低スコア応答の組をDPOデータに変換する。最後にQwenをLoRA/DPO学習し、小さいコーパスに近い会話戦略を再現できるか検証する。

## パイプライン

1. 小コーパス分析

```bash
python3 -m tools.analyze_small_corpus \
  --input data/small_corpus.jsonl \
  --output artifacts/bayes_models/generated_bayes_model.json
```

既定の分析モデルは `ANALYSIS_LLM_MODEL`、未設定時は `gpt-5.4-pro`。

2. 大規模対話データの評価

```bash
python3 -m tools.score_dialogue_with_bayes_model \
  --input data/large_dialogue.jsonl \
  --bayes-model artifacts/bayes_models/generated_bayes_model.json \
  --output artifacts/scored_dialogues/bayes_scored_dialogue.jsonl
```

既定の評価モデルは `SCORING_LLM_MODEL`、未設定時は `gpt-5.4`。

3. DPOデータ作成

```bash
python3 -m tools.build_dpo_from_bayes_scores \
  --input artifacts/scored_dialogues/bayes_scored_dialogue.jsonl \
  --output artifacts/datasets/bayes_dpo_preferences.jsonl
```

4. Qwen DPO LoRA学習

```bash
python3 -m tools.train_qwen35_dpo_lora \
  --dataset artifacts/datasets/bayes_dpo_preferences.jsonl \
  --model-id "${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}" \
  --output-dir artifacts/training_runs/qwen35_bayes_dpo_lora
```

## 入力仕様

小コーパス `JSONL` は1発話1行。

```json
{"conversation_id":"c001","turn_index":1,"speaker":"user","text":"昔はよく旅行しました。"}
{"conversation_id":"c001","turn_index":2,"speaker":"assistant","text":"どんな場所が特に印象に残っていますか。"}
```

大規模対話データ `JSONL` は1応答候補1行。

```json
{"conversation_id":"c001","turn_index":1,"prompt":"最近どうですか。","response":"その話をもう少し聞かせてください。"}
```

同じ `prompt` に複数の `response` があると、DPO変換時に高posterior応答を `chosen`、低posterior応答を `rejected` として組にできる。

## 生成ベイズモデル仕様

`tools.analyze_small_corpus` は次のキーを持つJSONを生成する。

```json
{
  "name": "target_style_model",
  "positive_state": "target_style",
  "negative_state": "non_target_style",
  "observations": ["deepening", "generic", "blocking"],
  "likelihoods": {
    "target_style": {"deepening": 0.7, "generic": 0.2, "blocking": 0.1},
    "non_target_style": {"deepening": 0.1, "generic": 0.3, "blocking": 0.6}
  },
  "prior": 0.5,
  "strategy_descriptions": {
    "deepening": "相手の内容を拾って自然に深める"
  }
}
```

`core.generated_bayes_model` は、観測ラベルの重複、尤度の不足、確率範囲、尤度合計を検証する。

## テスト

APIや実モデルを呼ばない軽量テスト。

```bash
python3 -m pytest \
  tests/test_generated_bayes_model.py \
  tests/test_bayes_research_pipeline.py \
  tests/test_local_llm_utils.py \
  tests/test_log_manager.py \
  tests/test_train_qwen35_dpo_lora.py \
  -v
```

GPUがない環境では、Qwenの実学習は `--dry-run` でデータ形式だけ確認する。
