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

状態遷移を持つベイズモデルを生成する場合:

```bash
python3 -m tools.analyze_small_corpus_transition_bayes \
  --input data/small_corpus.jsonl \
  --output artifacts/bayes_models/generated_transition_bayes_model.json
```

2. 大規模対話データの評価

```bash
python3 -m tools.score_dialogue_with_bayes_model \
  --input data/large_dialogue.jsonl \
  --bayes-model artifacts/bayes_models/generated_bayes_model.json \
  --output artifacts/scored_dialogues/bayes_scored_dialogue.jsonl
```

既定の評価モデルは `SCORING_LLM_MODEL`、未設定時は `gpt-5.4`。

状態遷移ベイズモデルで評価する場合:

```bash
python3 -m tools.score_dialogue_with_transition_bayes_model \
  --input data/large_dialogue.jsonl \
  --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json \
  --output artifacts/scored_dialogues/transition_bayes_scored_dialogue.jsonl
```

DailyDialogを使う場合は、まず文脈付き応答評価用JSONLへ変換する。

```bash
python3 -m tools.prepare_dailydialog_for_scoring \
  --split train \
  --max-dialogues 200 \
  --max-context-turns 8 \
  --output data/dailydialog_for_scoring_sample.jsonl
```

本番スコアリング前に、100〜300サンプルで `gpt-5.4` と `gpt-5.4-pro` の評価結果を比較する。

```bash
python3 -m tools.compare_scoring_models \
  --input data/dailydialog_for_scoring_sample.jsonl \
  --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json \
  --sample-size 200 \
  --output artifacts/scored_dialogues/scoring_model_comparison.json
```

差が小さい場合は、本番の大量スコアリングには `gpt-5.4` を使う。差が大きい場合のみ `gpt-5.4-pro` の利用を検討する。

高posterior応答を抽出する。

```bash
python3 -m tools.extract_high_posterior_dialogues \
  --input artifacts/scored_dialogues/dailydialog_transition_scored_sample.jsonl \
  --output artifacts/datasets/dailydialog_selected_en_sample.jsonl \
  --min-posterior 0.75 \
  --max-records 100 \
  --sort-by-posterior
```

抽出した英語応答を自然な日本語DPOデータへ変換する。

```bash
python3 -m tools.translate_and_generate_dpo \
  --input artifacts/datasets/dailydialog_selected_en_sample.jsonl \
  --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json \
  --output artifacts/datasets/dailydialog_ja_dpo_preferences_sample.jsonl \
  --model "${SCORING_LLM_MODEL:-gpt-5.4}" \
  --score-model "${SCORING_LLM_MODEL:-gpt-5.4}" \
  --candidates 4 \
  --min-score-gap 0.25
```

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

会話文脈を考慮する場合は、`prompt` に直前までの会話履歴を入れ、`response` に評価対象の次発話を入れる。

```json
{
  "conversation_id": "daily_000001",
  "turn_index": 5,
  "prompt": "user: 昔はよく旅行に行きました。\nassistant: どんな場所が印象に残っていますか。\nuser: 京都の桜が忘れられません。",
  "response": "その京都の桜は、どなたと見に行かれたんですか。"
}
```

この形式では、`turn_index=5` のスコアは「1〜4ターン目までの文脈に対する5ターン目の応答」の評価になる。つまり抽出単位は、会話全体ではなく、文脈付きの1応答である。

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

状態遷移版は `core.transition_bayes_model` が検証する。主な追加キーは次の通り。

```json
{
  "model_type": "transition_bayes_network",
  "states": ["opening", "deepening", "closing", "off_style"],
  "positive_states": ["deepening", "closing"],
  "negative_states": ["off_style"],
  "initial_state_prior": {"opening": 0.5, "deepening": 0.3, "closing": 0.1, "off_style": 0.1},
  "transition_likelihoods": {
    "opening": {"opening": 0.1, "deepening": 0.7, "closing": 0.1, "off_style": 0.1}
  },
  "emission_likelihoods": {
    "deepening": {"contextual_followup": 0.7, "generic_shift": 0.3}
  }
}
```

状態遷移版の `posterior` は、更新後の状態分布における `positive_states` の合計確率として出力する。

## スコアと抽出単位

スコアリング結果の `posterior` は、会話全体の最終評価ではなく、そのターンの応答を観測した直後のスコアである。

状態遷移版では、会話ごとに直前ターンの `state_posteriors` を保持し、次ターンの事前状態分布として使う。更新は次の流れになる。

```text
直前の状態分布
  → transition_likelihoods で次状態を予測
  → prompt/response から得た observation で更新
  → 新しい state_posteriors
  → positive_states の合計を posterior として出力
```

そのため、10ターンの会話で5ターン目の `posterior` が高い場合、現在の抽出対象は「1〜4ターン目までの文脈 + 5ターン目の応答」である。1〜5ターン目までの会話全体を丸ごと抽出する処理ではない。

DPO学習では、この文脈付き1応答を次の形で使う。

```json
{
  "prompt": "直前までの会話文脈",
  "chosen": "高posteriorの応答",
  "rejected": "低posteriorの応答",
  "metadata": {}
}
```

会話全体として良いデータを抽出したい場合は、別途 conversation-level score を作る。候補としては、平均 `posterior`、最終ターン `posterior`、低 `posterior` ターンの少なさ、`positive_states` 滞在率、ターンごとの `posterior` 推移などを使う。

## DailyDialogから日本語DPOを作る方針

DailyDialogは英語データなので、ベイズモデルで高posterior応答を抽出した後、日本人同士の自然な会話として日本語化する。翻訳は直訳ではなく、意図、感情、会話戦略、会話状態の流れを維持する。

最終JSONLは次のキーを持つ。

```json
{
  "prompt": "過去Nターンの日本語会話文脈",
  "chosen": "DailyDialogから抽出された高スコア応答を自然な日本語にしたもの",
  "rejected": "同じpromptに対する、一見自然だがベイズモデルでは低評価になる返答",
  "score_chosen": 0.9,
  "score_rejected": 0.3,
  "score_gap": 0.6,
  "source_dialogue_id": "train_000001",
  "turn_index": 5
}
```

可能な範囲で `translated_chosen`, `translated_rejected`, `state_sequence`, `strategy_sequence`, `reward_breakdown`, `translation_quality_score` も保持する。再現性のため、乱数シード、使用モデル、プロンプトテンプレート、ベイズモデルバージョンを `metadata` とmanifestに記録する。

`rejected` は文法的に破綻した返答や攻撃的な返答にはしない。複数候補を生成し、翻訳後chosenとrejected候補を同じ状態遷移ベイズモデルで再スコアリングする。`score_gap = score_chosen - score_rejected` が十分大きいサンプルを優先的に採用する。

モデル利用方針。

- `gpt-5.4-pro`: 小コーパス分析、ontology/state/strategy発見、状態遷移モデル設計、スコアリング基準設計、品質監査、失敗分析。
- `gpt-5.4`: DailyDialog全体の大量スコアリング、高スコア抽出、抽出後の日本語翻訳、rejected候補生成、再スコアリング。
- 本番前に100〜300サンプルで `gpt-5.4` と `gpt-5.4-pro` を比較し、スコア分布、上位一致率、順位相関、ベイズモデルとの整合性を見る。

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
