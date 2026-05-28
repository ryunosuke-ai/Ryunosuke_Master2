# Master2

小コーパスからLLMでベイズモデルを生成し、そのモデルで大規模対話データを評価してQwenのDPO学習データを作る研究用リポジトリです。

詳細は [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) を参照してください。

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
python3 -m pytest tests/test_generated_bayes_model.py tests/test_bayes_research_pipeline.py -v
```
