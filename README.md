# Master2

ESConvの小規模支援対話コーパスからLLMで状態遷移ベイズモデルを生成し、そのモデルで大規模対話候補を評価してQwenのDPO学習データを作る研究用リポジトリです。

詳細は [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) を参照してください。

現在の主実験は回想法ではなく、ESConvを用いた支援対話スタイル学習です。既存成果物の一部に `reminiscence_5000_to_2000` というRUN_TAGが残っていますが、これは過去の名前を引き継いだESConv実験成果物です。発表用の主要成果物は [artifacts/ESCONV_MANIFEST.md](artifacts/ESCONV_MANIFEST.md) に整理しています。

## 主な実験スクリプト

```bash
# ESConv Bayes-DPO学習
./scripts/run_dpo_pipeline_esconv_2000_watchdog.sh

# ESConv Oracle v3評価
./scripts/run_oracle_evaluation_esconv_v3_strategy.sh

# Bayes-DPO と Random-DPO のOracle比較
./scripts/run_oracle_evaluation_esconv_v3_strategy_bayes_vs_random.sh
```

## セットアップ

```bash
python3 -m venv test_env
source test_env/bin/activate
python3 -m pip install -r requirements.txt
```

`.env` には必要に応じて以下を設定します。

```env
OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_GPT56_ENDPOINT=
AZURE_OPENAI_GPT56_API_VERSION=
AZURE_OPENAI_GPT56_API_KEY=
AZURE_OPENAI_GPT56_SOL_DEPLOYMENT=gpt-5.6-sol
AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT=gpt-5.6-terra
LOCAL_QWEN_MODEL_ID=Qwen/Qwen3.5-27B
```

## 最小確認

```bash
python3 -B -m pytest -p no:cacheprovider -q
```

## MathDial前処理

公式Hugging Face revisionを固定し、train/test間で重複する`qid`を隔離して共通会話形式へ変換する。

```bash
python3 -m tools.prepare_mathdial \
  --config configs/datasets/mathdial.yaml \
  --output-root artifacts/mathdial_wildchat
```

書き出さずに取得・変換・統計だけを確認する場合:

```bash
python3 -m tools.prepare_mathdial --dry-run
```

Phase 0の既存コード調査結果は
[docs/BASIS_PIPELINE_REUSE_AUDIT.md](docs/BASIS_PIPELINE_REUSE_AUDIT.md)を参照する。

## MathDial × WildChat-1M

API/GPUを使用しない全stage接続確認:

```bash
./scripts/run_mathdial_wildchat_dry_run.sh
```

本実験はstage単位で停止・再開できる。

```bash
RUN_TAG=mathdial_wildchat_gpt56_v2 \
START_STAGE=build_basis END_STAGE=build_dpo \
WORKERS=8 \
./scripts/run_mathdial_wildchat_watchdog.sh
```

実行条件、成果物、ESConvからの再利用箇所は
[docs/MATHDIAL_WILDCHAT_PIPELINE.md](docs/MATHDIAL_WILDCHAT_PIPELINE.md)を参照する。
