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
ANALYSIS_LLM_MODEL=gpt-5.4-pro
SCORING_LLM_MODEL=gpt-5.4
LOCAL_QWEN_MODEL_ID=Qwen/Qwen3.5-27B
```

## 最小確認

```bash
python3 -B -m pytest -p no:cacheprovider -q
```
