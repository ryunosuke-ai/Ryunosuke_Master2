# ESConv発表用成果物manifest

このmanifestは、発表準備で削除してはいけないESConv支援対話スタイル学習実験の主要成果物を整理するためのものです。名前に `reminiscence` が含まれる成果物がありますが、現在の研究対象は回想法ではなくESConvです。

## Bayes-DPO本命実験

- RUN_TAG: `reminiscence_5000_to_2000`
- 実験内容: ESConv小コーパスから生成した状態遷移ベイズモデルでDailyDialog候補を評価し、DailyDialog DPO 2000件とESConv gold DPO 500件を混合してQwen3.5-27BをDPO LoRA学習した実験。
- Bayes model: `artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json`
- DPO dataset: `artifacts/datasets/esconv_mixed_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl`
- DailyDialog DPO source: `artifacts/datasets/dailydialog_ja_dpo_preferences_reminiscence_5000_to_2000_daily.jsonl`
- ESConv gold DPO source: `artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl`
- LoRA adapter: `artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit`
- Chunk logs: `artifacts/run_logs/reminiscence_5000_to_2000/chunks/`

## Oracle評価

- ESConv v3 strategy評価: `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy`
- ESConv v3 strategy評価、GPT-5.4 Oracle: `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54`
- Bayes-DPO vs Random-DPO評価: `artifacts/evaluations/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy`
- 参考用v2評価: `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v2`

## Random-DPO baseline

- RUN_TAG: `esconv_5000_to_2000_random2500`
- DPO dataset: `artifacts/datasets/dailydialog_random2500_ja_dpo_preferences_esconv_5000_to_2000_random2500.jsonl`
- DailyDialog source: `artifacts/datasets/dailydialog_ja_dpo_preferences_random2500_esconv_5000_to_2000_random2500_daily.jsonl`
- Manifest: `artifacts/datasets/dailydialog_random2500_ja_dpo_preferences_esconv_5000_to_2000_random2500.manifest.json`
- LoRA adapter: `artifacts/training_runs/qwen35_random2500_dailydialog_dpo_lora_esconv_5000_to_2000_random2500_ep1_lr5e-6_r8_a16_no4bit`

## 削除しない方針

- 上記のBayesモデル、DPOデータ、LoRA adapter、Oracle評価結果は削除しない。
- `reminiscence` を含むパスでも、ESConv成果物としてこのmanifestに載っているものは削除しない。
- パスのリネームは発表前には行わず、必要な場合はREADMEやmanifestで意味を補足する。

## 手動確認後に整理する候補

- `logs/`: 実行ログ。発表用の根拠確認が終わるまで保持する。
- `artifacts/run_logs/*.heartbeat.json`: 長時間実行のheartbeat。再現性の根拠として不要なら削除可能。
- `data/esconv_analysis_corpus_reminiscence_5000_to_2000DPO_WORKERS=4.jsonl` と `artifacts/run_logs/reminiscence_5000_to_2000DPO_WORKERS=4/`: RUN_TAGに環境変数文字列が混入した可能性があるため、参照ゼロと内容重複を確認してから削除する。
