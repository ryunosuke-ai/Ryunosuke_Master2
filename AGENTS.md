# Repository Guidelines

## このリポジトリについて

このリポジトリは、小さい会話コーパスからLLMで会話特徴・会話戦略・観測ラベルを分析し、そのコーパスらしさを表すベイズモデルを自動生成する研究用リポジトリです。生成したベイズモデルを使って大きな対話データを評価し、DPO用の preference データを作成して、Qwen をLoRA/DPO学習します。

目標は「小さいコーパスに見られる会話の進め方や応答戦略を、ローカルLLMが再現できるか」を検証することです。

## 研究の全体像

処理の流れは次の通りです。

```text
小コーパスJSONL
  → GPT-5.4-pro による会話特徴・戦略分析
  → 生成ベイズモデルJSON
  → GPT-5.4 による大規模対話データの観測ラベル評価
  → posterior による応答スコアリング
  → DPO preference JSONL
  → Qwen のLoRA/DPO学習
  → ベースQwenとDPO後Qwenの比較
```

既定の役割分担:

- 小コーパス分析LLM: `gpt-5.4-pro`
- 大規模対話データ評価LLM: `gpt-5.4`
- 学習対象ローカルLLM: `Qwen/Qwen3.5-27B`

モデル名はコードに固定せず、環境変数で差し替え可能にしてください。

## 主要ファイル

- `tools/analyze_small_corpus.py`: 小コーパスを分析し、ベイズモデルJSONを生成する
- `core/generated_bayes_model.py`: 生成ベイズモデルJSONの読み込み、検証、posterior更新を行う
- `tools/score_dialogue_with_bayes_model.py`: 大規模対話データを生成ベイズモデルでスコアリングする
- `tools/build_dpo_from_bayes_scores.py`: スコア済み対話からDPO preference JSONLを作る
- `tools/train_qwen35_dpo_lora.py`: QwenをDPO LoRA学習する
- `apps/dpo_base_chat.py`: ベースQwenとの単独チャット
- `apps/dpo_trained_chat.py`: DPO後Qwenとの単独チャット
- `apps/dpo_compare_chat.py`: ベースQwenとDPO後QwenのStreamlit比較UI
- `apps/dpo_compare_text_chat.py`: ベースQwenとDPO後Qwenのターミナル比較CLI
- `apps/dpo_log_viewer.py`: DPO比較ログの閲覧UI
- `docs/RESEARCH_PLAN.md`: 研究計画と入出力仕様の詳細

## 入力データ仕様

小コーパスは、1発話1行のJSONLを基本形式にします。

```json
{"conversation_id":"c001","turn_index":1,"speaker":"user","text":"昔はよく旅行しました。"}
{"conversation_id":"c001","turn_index":2,"speaker":"assistant","text":"どんな場所が特に印象に残っていますか。"}
```

大規模対話データは、1応答候補1行のJSONLを基本形式にします。

```json
{"conversation_id":"c001","turn_index":1,"prompt":"最近どうですか。","response":"その話をもう少し聞かせてください。"}
```

同じ `prompt` に複数の `response` があると、DPO変換時に高posterior応答を `chosen`、低posterior応答を `rejected` として組にできます。

## 生成ベイズモデル仕様

生成ベイズモデルJSONは、少なくとも次のキーを持ちます。

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

`core/generated_bayes_model.py` は、観測ラベルの重複、尤度の不足、確率範囲、尤度合計を検証します。ベイズ更新の中心ロジックは、このモジュールに集約してください。

## 起動・実行例

小コーパス分析:

```bash
python3 -m tools.analyze_small_corpus \
  --input data/small_corpus.jsonl \
  --output artifacts/bayes_models/generated_bayes_model.json
```

大規模対話データの評価:

```bash
python3 -m tools.score_dialogue_with_bayes_model \
  --input data/large_dialogue.jsonl \
  --bayes-model artifacts/bayes_models/generated_bayes_model.json \
  --output artifacts/scored_dialogues/bayes_scored_dialogue.jsonl
```

DPOデータ作成:

```bash
python3 -m tools.build_dpo_from_bayes_scores \
  --input artifacts/scored_dialogues/bayes_scored_dialogue.jsonl \
  --output artifacts/datasets/bayes_dpo_preferences.jsonl
```

Qwen DPO LoRA学習:

```bash
python3 -m tools.train_qwen35_dpo_lora \
  --dataset artifacts/datasets/bayes_dpo_preferences.jsonl \
  --model-id "${LOCAL_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}" \
  --output-dir artifacts/training_runs/qwen35_bayes_dpo_lora
```

## 環境変数

`.env` には必要に応じて以下を設定します。

```env
OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
ANALYSIS_LLM_MODEL=gpt-5.4-pro
SCORING_LLM_MODEL=gpt-5.4
LOCAL_QWEN_MODEL_ID=Qwen/Qwen3.5-27B
```

Azure OpenAIを使う場合は、deployment名を `ANALYSIS_LLM_MODEL` や `SCORING_LLM_MODEL` に入れる前提です。

## テスト

APIや実モデルを呼ばない軽量テストを優先してください。

```bash
python3 -B -m pytest -p no:cacheprovider -v
```

GPUがない環境では、Qwenの実学習は行わず、`tools/train_qwen35_dpo_lora.py --dry-run` でデータ形式だけ確認してください。

## コーディング方針

- PythonはPEP 8準拠、インデントは4スペース
- 関数・変数・ファイル名は `snake_case`
- クラス名は `PascalCase`
- コメント、エラーメッセージ説明、ドキュメントは日本語で書く
- API呼び出し部分はテストで差し替えられるよう、できるだけ小さなインターフェースに閉じ込める
- ベイズモデルの検証・更新処理は `core/generated_bayes_model.py` に集約する
- DPO JSONLは `prompt`, `chosen`, `rejected`, `metadata` の形を維持する

## コピーしない・コミットしないもの

以下は原則としてコミットしないでください。

- `.env`, `.env.*`
- APIキー、個人情報、会話参加者を特定できる生ログ
- `data/`, `datasets/`, `logs/`
- `artifacts/bayes_models/`
- `artifacts/scored_dialogues/`
- `artifacts/datasets/`
- `artifacts/training_runs/`
- `models/`, `hf_cache/`
- 仮想環境、キャッシュ、`__pycache__/`

生成物は再現手順をドキュメントに残し、必要な場合だけ別管理してください。

## 作業ルール

- このリポジトリの保存先は `git@github.com:ryunosuke-ai/Ryunosuke_Master2.git`
- 旧リポジトリは `old-origin` として残し、新しい `origin` へpushする
- 通常の作業ブランチは `master`
- 変更後は関連テストを実行し、commitし、新しい `origin` へpushする
- push前には必ず `git status --short`, `git diff --stat`, `git diff --cached --stat` を確認する
- `git add .` は使わず、stageは明示ファイルのみ行う
- force pushは禁止。push拒否や認証エラーが出たら停止して報告する
- コミットメッセージは次の形式にする
  - `feat: ○○機能を追加`
  - `fix: ○○のバグを修正`
  - `refactor: ○○をリファクタリング`
  - `docs: ○○のドキュメントを更新`
  - `chore: ○○`
- `.env`、APIキー、個人情報を含むログ、大容量データ、学習済みモデル成果物、生JSONLはコミット・push禁止
- `logs/`, `artifacts/run_logs/`, `artifacts/training_runs/`, `artifacts/datasets/`, `artifacts/evaluations/`, `artifacts/bayes_models/`, `hf_cache/` は原則pushしない。発表用評価結果は軽量な `summary.json` と `manifest.json` だけを `docs/results/` へコピーして管理する

## 旧Masterとの関係

旧 `Master` は、高齢者との自然な対話を目指すマルチモーダル会話エージェントの研究リポジトリでした。`Master2` では、その音声I/O、画像会話フェーズ制御、手書きの会話フェーズ管理は主目的ではありません。

このリポジトリでは、旧研究からDPO/Qwen/ログ基盤だけを軽量に流用し、研究対象を「LLMによる小コーパス分析、ベイズモデル自動生成、ベイズスコアによるDPO学習」に絞ります。
