# 発表用軽量評価結果

このディレクトリには、発表準備で参照する軽量な評価結果だけを置く。

完全なOracle評価成果物は `artifacts/evaluations/` 配下にあり、応答本文や判定ログを含むためGit管理しない。Git管理するのは各runの `summary.json` と `manifest.json` のみとする。

## 収録結果

| Git管理パス | 元artifactパス | 内容 |
|---|---|---|
| `docs/results/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/` | `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/` | ESConv Bayes-DPOのOracle v3 strategy評価 |
| `docs/results/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/` | `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/` | ESConv Bayes-DPOのGPT-5.4 Oracle v3 strategy評価 |
| `docs/results/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/` | `artifacts/evaluations/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/` | ESConv Bayes-DPOとRandom-DPOのOracle v3 strategy比較 |

## Git管理しないもの

- `responses.jsonl`
- `judgments.jsonl`
- `summary.partial.json`
- LoRA adapter、モデル重み、DPO JSONL、生ログ、run_logs、cache

`reminiscence_5000_to_2000` を含む名前は過去RUN_TAGを引き継いでいるが、ここに収録した結果の実体はESConv支援対話スタイル学習実験である。
