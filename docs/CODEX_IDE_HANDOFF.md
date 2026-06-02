# Codex IDE 引き継ぎメモ

このファイルは、Codex CLI で進めてきた会話・設計・実装内容を、Codex IDE へ移行しても追えるようにまとめたものです。

## 現在の研究目的

この研究の主眼は、手作業でベイズモデルを設計することではなく、小さい高品質な会話コーパスをLLMに分析させ、そのコーパスらしい会話特徴・応答戦略・状態遷移を表すベイズモデルを自動生成することです。

全体の流れは次の通りです。

```text
小さい高品質会話コーパス
  → GPT-5.4-pro によるコーパス目的・会話戦略・状態遷移の推定
  → 状態遷移ベイズモデルJSONの自動生成
  → GPT-5.4 による大規模対話データの観測ラベル評価
  → ベイズ更新による posterior スコア計算
  → 高posterior応答を chosen、低posterior応答を rejected としてDPOデータ化
  → Qwen のLoRA/DPO学習
```

研究上の重点は、生成したベイズモデル自体の厳密な精度検証よりも、そのモデルを使って大規模会話データから小コーパスに合った会話応答を抽出することにある。

## 重要な設計判断

小コーパスは回想法ベースの会話例だが、GPT-5.4-proへ渡すプロンプトでは「回想法」と明示しない。これは、LLMが小コーパス自体から「このデータセットが何を良い会話としているのか」を推定できるかを見たいからである。

そのため、生成プロンプトでは次を明示している。

- データセット目的や会話スタイルは、事前知識で決めつけずコーパスから推定する。
- 小コーパスらしい応答を高く評価する。
- 小コーパスから外れる応答を低く評価する。
- 大量の `prompt/response` 評価、posterior による抽出、DPO preference 作成に使う。

最初に作った `tools/analyze_small_corpus.py` は、2状態の単純なベイズモデルを生成する。これは有用だが、会話文脈と状態遷移を表しにくい。

その後、会話文脈を考慮する目的で、別系統として状態遷移ベイズモデルを生成する `tools/analyze_small_corpus_transition_bayes.py` を追加した。今後の本命はこの状態遷移版である。

## 実装済みファイル

### 2状態ベイズモデル系

- `tools/analyze_small_corpus.py`
  - 小コーパスから2状態ベイズモデルJSONを生成する。
  - `target_style` と `non_target_style` の2状態。
  - `observations` と `likelihoods` を生成する。
  - GPT-5.4-pro用のAzure/OpenAI環境変数に対応済み。
  - JSONモードを使い、出力前に `parse_bayes_model()` で検証する。

- `core/generated_bayes_model.py`
  - 2状態ベイズモデルの読み込み、検証、posterior更新を行う。

- `tools/score_dialogue_with_bayes_model.py`
  - 大規模対話データを2状態ベイズモデルで評価する。

### 状態遷移ベイズモデル系

- `tools/analyze_small_corpus_transition_bayes.py`
  - 小コーパスから状態遷移ベイズモデルJSONを生成する。
  - `states`, `transition_likelihoods`, `emission_likelihoods` を持つ。
  - 出力は `model_type = "transition_bayes_network"`。
  - 既定の `--max-output-tokens` は `20000`。
  - 実行中に経過秒つきの進捗を表示する。
  - JSON構文が壊れた場合は、1回だけJSON修復プロンプトを呼ぶ。

- `core/transition_bayes_model.py`
  - 状態遷移ベイズモデルの読み込み、検証、更新式を実装している。
  - 中心関数は次の通り。
    - `parse_transition_bayes_model()`
    - `predict_next_state_distribution()`
    - `update_state_distribution()`
    - `positive_posterior()`
    - `score_transition_observation()`

- `tools/score_dialogue_with_transition_bayes_model.py`
  - 状態遷移ベイズモデルで大規模対話データを評価する。
  - 会話ごとに直前ターンの `state_posteriors` を保持し、次ターンの事前分布として使う。

- `tests/test_transition_bayes_model.py`
  - 状態遷移ベイズモデルの検証、更新式、生成、スコアリングの軽量テスト。

## 実行済みの主なコマンド

小コーパスのdry-run。

```bash
python3 -m tools.analyze_small_corpus \
  --input data/small_corpus.jsonl \
  --output artifacts/bayes_models/generated_bayes_model.json \
  --dry-run
```

このときの概要は次の通り。

```text
records: 80
conversations: 10
speakers: 2
max_text_chars: 51
model: gpt-5.4-pro
```

2状態ベイズモデル生成。

```bash
python3 -m tools.analyze_small_corpus \
  --input data/small_corpus.jsonl \
  --output artifacts/bayes_models/generated_bayes_model.json
```

状態遷移ベイズモデル生成。

```bash
python3 -m tools.analyze_small_corpus_transition_bayes \
  --input data/small_corpus.jsonl \
  --output artifacts/bayes_models/generated_transition_bayes_model.json
```

生成済みの状態遷移ベイズモデル。

```text
artifacts/bayes_models/generated_transition_bayes_model.json
```

JSONを見やすく表示するコマンド。

```bash
python3 -m json.tool artifacts/bayes_models/generated_transition_bayes_model.json
```

状態遷移モデルで大規模データを評価する予定のコマンド。

```bash
python3 -m tools.score_dialogue_with_transition_bayes_model \
  --input data/large_dialogue.jsonl \
  --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json \
  --output artifacts/scored_dialogues/transition_bayes_scored_dialogue.jsonl
```

DPO preference作成予定のコマンド。

```bash
python3 -m tools.build_dpo_from_bayes_scores \
  --input artifacts/scored_dialogues/transition_bayes_scored_dialogue.jsonl \
  --output artifacts/datasets/transition_bayes_dpo_preferences.jsonl
```

## APIまわりで解決した問題

最初は `tools/analyze_small_corpus.py` のAzure OpenAI呼び出しで `401` が出た。原因候補として、環境変数名やdeployment名の参照ずれがあった。

同じAPIを使う `apps/openai_gpt54_compare_chat.py` は正常に動いていたため、その実装を参考にして、分析用モデル・APIキーの解決を修正した。

現在は以下のような環境変数フォールバックに対応している。

- `ANALYSIS_LLM_MODEL`
- `AZURE_OPENAI_GPT54_PRO_DEPLOYMENT_NAME`
- `OPENAI_GPT54_PRO_MODEL`
- `AZURE_OPENAI_GPT54_PRO_API_KEY`
- `OPENAI_GPT54_PRO_API_KEY`
- `AZURE_OPENAI_API_KEY`

また、Responses APIのJSONモード使用時に、

```text
Response input messages must contain the word 'json'
```

という400エラーが出たため、JSONモード時はinput側にも `Return a valid JSON object only.` を付与するようにした。

状態遷移モデルでは出力が長くなり、`max_output_tokens` で途中終了したため、状態遷移版の既定値を `20000` に上げた。また、プロンプト内の例JSONを短くし、状態数・観測数を `3〜6個` に絞った。

## 生成済み状態遷移ベイズモデルの内容

現在の生成ファイルは次。

```text
artifacts/bayes_models/generated_transition_bayes_model.json
```

モデル名。

```text
reminiscence_support_transition_model
```

GPT-5.4-proが推定したデータセット仮説。

```text
過去の生活・家族・仕事・季節行事の思い出を、共感的な受け止めと具体的な追想質問で引き出し、最後に温かく要約して終える回想支援型対話コーパス。
```

重要なのは、プロンプトでは「回想法」と明示していないのに、生成モデルが小コーパスから回想支援型対話だと推定している点である。これは今回の研究目的にかなり合っている。

### states

生成された状態は5つ。

```json
[
  "opening_invitation",
  "setting_sensory_detail",
  "activity_social_detail",
  "warm_closure",
  "off_style"
]
```

各状態の意味。

- `opening_invitation`: 相手の思い出をやさしく受け止め、最初の具体的内容を尋ねる状態。
- `setting_sensory_detail`: 景色、音、匂い、季節、場所の雰囲気など、情景・感覚を深掘りする状態。
- `activity_social_detail`: 誰といたか、何をしていたか、家族・友人・仕事仲間・日課など、人間関係や活動を深掘りする状態。
- `warm_closure`: 聞いた内容を温かくまとめ、懐かしさや大切さを認めて締める状態。
- `off_style`: 助言、説明、話題逸脱、事務的応答など、小コーパスらしい聞き手から外れた状態。

### positive_states / negative_states

```json
"positive_states": [
  "opening_invitation",
  "setting_sensory_detail",
  "activity_social_detail",
  "warm_closure"
],
"negative_states": [
  "off_style"
]
```

`off_style` 以外の4状態を、小コーパスらしい良い会話状態として扱う。スコアリング時の `posterior` は、更新後の状態分布における `positive_states` の合計である。

### observations

生成された観測ラベルは5つ。

```json
[
  "ack_open_probe",
  "sensory_setting_focus",
  "activity_social_focus",
  "warm_summary_close",
  "generic_or_unrelated"
]
```

各観測の意味。

- `ack_open_probe`: 相手の発話を受け止めつつ、最初の具体的な追想を促す応答。
- `sensory_setting_focus`: 景色、音、匂い、季節、部屋や町の雰囲気など、情景・感覚に注目する応答。
- `activity_social_focus`: 一緒にいた人、家族、友人、仕事仲間、日課、会話内容など、行動や人間関係に注目する応答。
- `warm_summary_close`: 聞いた内容を温かくまとめて締める応答。
- `generic_or_unrelated`: 一般論、助言、説明、話題変更、短すぎる返答など、思い出を具体的に深めない応答。

### initial_state_prior

会話開始時の状態分布。

```json
{
  "opening_invitation": 0.84,
  "setting_sensory_detail": 0.06,
  "activity_social_detail": 0.04,
  "warm_closure": 0.03,
  "off_style": 0.03
}
```

会話開始時は `opening_invitation` から始まる可能性が高い、という設計になっている。

### transition_likelihoods の解釈

`transition_likelihoods` は `P(next_state | current_state)` である。

例。

```json
"opening_invitation": {
  "setting_sensory_detail": 0.56,
  "activity_social_detail": 0.24,
  "warm_closure": 0.09,
  "opening_invitation": 0.06,
  "off_style": 0.05
}
```

これは、`opening_invitation` の次は `setting_sensory_detail` に進みやすい、という意味である。つまり、相手の思い出を聞き出したあと、情景や感覚の深掘りへ進む流れが強い。

他にも重要な遷移は次。

- `setting_sensory_detail → activity_social_detail`: 0.43
- `activity_social_detail → warm_closure`: 0.50
- `warm_closure → warm_closure`: 0.85
- `off_style → off_style`: 0.60

このため、モデル全体としては次のような会話進行を仮定している。

```text
opening_invitation
  → setting_sensory_detail
  → activity_social_detail
  → warm_closure
```

### emission_likelihoods の解釈

`emission_likelihoods` は `P(observation | state)` である。

例。

```json
"setting_sensory_detail": {
  "sensory_setting_focus": 0.78
}
```

これは、状態が `setting_sensory_detail` のとき、`sensory_setting_focus` という観測が出やすい、という意味である。

他にも対応がかなり自然。

- `opening_invitation` では `ack_open_probe`: 0.60
- `setting_sensory_detail` では `sensory_setting_focus`: 0.78
- `activity_social_detail` では `activity_social_focus`: 0.73
- `warm_closure` では `warm_summary_close`: 0.85
- `off_style` では `generic_or_unrelated`: 0.83

## ベイズ更新式

状態遷移版では、各ターンで次の2段階を行う。

### 1. 遷移で次状態を予測

直前の状態分布を `P(s_t)` とすると、次ターン前の予測分布は次。

```text
P(s_{t+1}) = Σ P(s_t) * P(s_{t+1} | s_t)
```

これは、直前の状態分布だけを使って次状態を予測するマルコフ的な更新である。

### 2. 観測ラベルでベイズ更新

GPT-5.4が `prompt/response` から観測ラベル `o_{t+1}` を判定したら、各状態について次を計算する。

```text
未正規化スコア = P(s_{t+1}) * P(o_{t+1} | s_{t+1})
```

その後、合計が1になるように正規化する。

```text
P(s_{t+1} | o_{t+1})
= P(o_{t+1} | s_{t+1}) * P(s_{t+1})
  / Σ_s P(o_{t+1} | s) * P(s)
```

最後に、`positive_states` の合計を `posterior` として出す。

```text
posterior = Σ P(state | observation)
            for state in positive_states
```

## 具体的な更新例

会話開始時の `initial_state_prior` から1ターン進めると、観測前の予測分布は次のようになる。

```text
opening_invitation:      0.0583
setting_sensory_detail:  0.5012
activity_social_detail:  0.2382
warm_closure:            0.1355
off_style:               0.0668
```

ここでGPT-5.4が `prompt/response` を見て、

```text
observation = sensory_setting_focus
```

と判定した場合、各状態の重みは次。

```text
opening_invitation:
0.0583 * 0.17 = 0.00991

setting_sensory_detail:
0.5012 * 0.78 = 0.39094

activity_social_detail:
0.2382 * 0.08 = 0.01906

warm_closure:
0.1355 * 0.03 = 0.00407

off_style:
0.0668 * 0.04 = 0.00267
```

正規化後の状態分布は次。

```text
opening_invitation:      0.0232
setting_sensory_detail:  0.9163
activity_social_detail:  0.0447
warm_closure:            0.0095
off_style:               0.0063
```

`positive_states` の合計は次。

```text
0.0232 + 0.9163 + 0.0447 + 0.0095 = 0.9937
```

つまり、このターンの `posterior` は約 `0.9937` であり、小コーパスらしい応答として非常に高く評価される。

逆に、観測が `generic_or_unrelated` の場合は次のようになる。

```text
opening_invitation:      0.0200
setting_sensory_detail:  0.1718
activity_social_detail:  0.0817
warm_closure:            0.0929
off_style:               0.6336
```

このときの `posterior` は約 `0.3664`。`off_style` に強く寄るため、スコアが低くなる。

## posterior の意味

現在の実装における `posterior` は、会話全体の最終スコアではない。各ターンで、その応答を観測した直後に「小コーパスらしい状態にどれだけいるか」を表すスコアである。

例えば10ターンの会話で5ターン目の `posterior` が高い場合、現在の抽出対象は次。

```text
1〜4ターン目までの会話文脈 + 5ターン目の応答
```

1〜5ターン目までの会話全体を丸ごと抽出しているわけではない。

会話全体を抽出したい場合は、別途 conversation-level score を作る必要がある。候補は次。

- 平均 `posterior`
- 最終ターンの `posterior`
- 低 `posterior` ターンの少なさ
- `positive_states` 滞在率
- `posterior` 推移の安定性

## 大規模データセット候補

当初は Gutenberg Dialogue Dataset を候補にしていたが、会話文脈付きの複数ターン対話を大量に扱う目的では、DailyDialog の方が合っていそうだと判断した。

理由。

- 複数ターン会話が含まれる。
- 日常会話として自然な文脈を持つ。
- `prompt` に直前までの文脈、`response` に次発話を入れる形に変換しやすい。
- 大量データから小コーパスらしい応答を抽出する研究目的に合いやすい。

大規模データの1行は次の形にする。

```json
{
  "conversation_id": "daily_000001",
  "turn_index": 5,
  "prompt": "user: ...\nassistant: ...\nuser: ...",
  "response": "..."
}
```

このとき `prompt` には評価対象応答の直前までの会話文脈を入れる。`response` には評価対象となる次の発話だけを入れる。

## 現在の評価

生成済み状態遷移ベイズモデルは、今回の自作回想法ベース小コーパスをかなりよく分析できていると見ている。

良い点。

- プロンプトで「回想法」と明示していないのに、回想支援型対話だと推定できている。
- 状態設計が `導入 → 情景・感覚 → 活動・人間関係 → 温かい締め` になっている。
- `off_style` が助言、説明、話題逸脱、事務的応答などとして明確に定義されている。
- `transition_likelihoods` が自然な会話進行を表している。
- `emission_likelihoods` が状態と観測ラベルの対応として分かりやすい。

注意点。

- 生成された確率値は、実データから頻度推定した厳密な統計量ではない。
- GPT-5.4-proが小コーパスを読んで作った「仮説的な確率」である。
- 研究上は「小コーパスからLLMにより生成された仮説的な状態遷移ベイズモデル」と表現するのが正確。

## これまでに通ったテスト

関連テストは次で通っている。

```bash
python3 -B -m pytest -p no:cacheprovider -v \
  tests/test_transition_bayes_model.py \
  tests/test_bayes_research_pipeline.py
```

結果。

```text
23 passed
```

以前、全体テストも通っている。

```bash
python3 -B -m pytest -p no:cacheprovider -v
```

結果。

```text
105 passed, 2 warnings
```

警告はSWIG由来のdeprecation warningで、今回の変更とは無関係。

## 次にやるとよいこと

1. DailyDialogを大規模データとして用意する。
2. DailyDialogを `conversation_id`, `turn_index`, `prompt`, `response` のJSONLに変換するスクリプトを作る。
3. `prompt` には直前までの会話文脈を入れる。
4. `response` には次発話を入れる。
5. `tools.score_dialogue_with_transition_bayes_model` でスコアリングする。
6. 高posterior応答と低posterior応答を少数サンプルで目視確認する。
7. 問題なさそうならDPO preference JSONLを作る。
8. QwenのDPO LoRA学習へ進む。

優先度が高い実装候補。

- `tools/prepare_dailydialog_for_scoring.py`
  - DailyDialogを状態遷移スコアリング用JSONLへ変換する。
  - 直前Nターンを `prompt` に入れる。
  - 次発話を `response` に入れる。

- 会話全体スコア集計ツール
  - ターン単位 `posterior` を会話単位へ集約する。
  - 平均posterior、最終posterior、低posterior率、positive状態滞在率などを出す。

## IDE移行時の最初の確認ポイント

Codex IDEで再開したら、まず次を確認するとよい。

```bash
git status --short
```

この時点では、実装途中ではなく、状態遷移ベイズモデル生成・スコアリング系の追加とドキュメント更新が入っている状態。

重要な生成物は `artifacts/` 配下にあるが、`artifacts/bayes_models/` は原則コミットしない。生成手順をドキュメントに残し、必要に応じてローカルで再生成する。

`.env` やAPIキーは絶対にコミットしない。

