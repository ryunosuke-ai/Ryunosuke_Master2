# ESConv発表前リポジトリ整理監査

作成日: 2026-06-10

現在の主実験は回想法ではなく、ESConvを用いた支援対話スタイル学習実験である。`reminiscence_5000_to_2000` を含む成果物名は過去RUN_TAGを引き継いだもので、manifestに記載したものは削除しない。

## 1. ESConv実験で現在も必要なファイル

| 種別 | パス | 理由 |
|---|---|---|
| ツール | `tools/prepare_esconv_for_analysis.py` | ESConv小コーパス作成 |
| ツール | `tools/analyze_esconv_corpus_transition_bayes.py` | ESConv状態遷移ベイズモデル生成 |
| ツール | `tools/build_esconv_gold_dpo.py` | ESConv gold DPO生成 |
| ツール | `tools/run_oracle_evaluation.py` | base vs DPO Oracle評価 |
| ツール | `tools/run_oracle_evaluation_lora_pair.py` | Bayes-DPO vs Random-DPO Oracle評価 |
| スクリプト | `scripts/run_dpo_pipeline_esconv_2000_chunked.sh` | ESConv Bayes-DPO本体 |
| スクリプト | `scripts/run_dpo_pipeline_esconv_2000_watchdog.sh` | 長時間実行監視 |
| スクリプト | `scripts/run_oracle_evaluation_esconv_v3_strategy.sh` | ESConv v3評価 |
| スクリプト | `scripts/run_oracle_evaluation_esconv_v3_strategy_bayes_vs_random.sh` | Bayes-DPOとRandom-DPO比較 |
| 設定 | `configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl` | 発表用Oracle prompt |
| テスト | `tests/test_esconv_pipeline.py`, `tests/test_oracle_evaluation.py`, `tests/test_random_dpo_baseline.py` | ESConv/Oracle/Random-DPOの軽量検証 |

## 2. 回想法実験に由来し、現在は不要と思われるファイル

| パス | 判断 |
|---|---|
| `scripts/run_dpo_pipeline_reminiscence_2000.sh` | 旧回想法再現用。発表前は削除せず非推奨表示を追加済み |
| `scripts/run_dpo_pipeline_reminiscence_2000_watchdog.sh` | 旧回想法再現用。発表前は削除せず非推奨表示を追加済み |
| `scripts/run_oracle_evaluation_reminiscence.sh` | 旧回想法Oracle用。発表前は削除せず非推奨表示を追加済み |
| `configs/evaluation_prompts/reminiscence_oracle_eval_v1.jsonl` | 旧評価prompt。削除は手動確認後 |
| `configs/evaluation_prompts/reminiscence_oracle_eval_v2_100.jsonl` | 旧評価prompt。削除は手動確認後 |
| `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_v1/` | 旧評価結果。削除は発表資料で不要確認後 |

## 3. `reminiscence` を含むが削除してはいけないESConv成果物

| パス | 実体 |
|---|---|
| `artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit` | ESConv Bayes-DPO LoRA |
| `artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json` | ESConv transition Bayes model |
| `artifacts/datasets/esconv_mixed_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl` | ESConv mixed DPO dataset |
| `artifacts/datasets/dailydialog_ja_dpo_preferences_reminiscence_5000_to_2000_daily.jsonl` | DailyDialog由来DPO source |
| `artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl` | ESConv gold DPO source |
| `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/` | ESConv v3 Oracle評価 |
| `artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/` | ESConv v3 GPT-5.4 Oracle評価 |
| `data/esconv_analysis_corpus_reminiscence_5000_to_2000.jsonl` | ESConv分析用小コーパス |
| `artifacts/run_logs/reminiscence_5000_to_2000/chunks/` | ESConv DPO生成チャンクログ |

## 4. git管理や発表準備には不要と思われるログ・中間成果物・キャッシュ

| パス | 判断 |
|---|---|
| `__pycache__/`, `.pytest_cache/` | 削除済み |
| `logs/` | git管理不要。発表用の根拠確認が終わるまで保持 |
| `hf_cache/` | git管理不要。再取得可能だが、発表直前は削除しない |
| `artifacts/run_logs/*.heartbeat.json` | 再実行監視用。必要なければ手動削除可能 |
| `artifacts/run_logs/reminiscence_5000_to_2000DPO_WORKERS=4/` | RUN_TAG混入の疑い。参照ゼロと重複確認後に削除候補 |
| `data/esconv_analysis_corpus_reminiscence_5000_to_2000DPO_WORKERS=4.jsonl` | RUN_TAG混入の疑い。参照ゼロと重複確認後に削除候補 |

## 5. リネームした方がよいが、発表前は実パスを変えないもの

| 現在名 | 理想名 | 対応 |
|---|---|---|
| `reminiscence_5000_to_2000` | `esconv_5000_to_2000` | 成果物はリネームせずmanifestで説明 |
| `qwen35_bayes_dpo_lora_reminiscence_5000_to_2000...` | `qwen35_esconv_bayes_dpo_lora_5000_to_2000...` | LoRAパスは変更しない |
| `scripts/run_dpo_pipeline_reminiscence_2000.sh` | なし | 旧再現用として非推奨表示 |
| `scripts/run_esconv_then_reminiscence_tail.sh` | なし | 旧複合実行用として非推奨表示 |

## 6. リネームで参照が壊れる可能性がある箇所

| 対象 | 壊れる参照 |
|---|---|
| Bayes-DPO LoRAディレクトリ | チャットアプリ既定値、Oracleスクリプト、既存評価manifest |
| Bayes model JSON | Oracle評価、DPO生成manifest、評価再計算 |
| DPO dataset JSONL | 学習再現、DPO監査、成果物manifest |
| Oracle評価ディレクトリ | 発表用summary参照、`docs/CODEX_RESEARCH_HANDOFF_CURRENT.md` |
| `RUN_TAG` | Bayesモデル、DPOデータ、LoRA、評価出力の対応関係 |

## 実施済み整理

- ESConv発表用成果物manifestを `artifacts/ESCONV_MANIFEST.md` に追加。
- README、研究計画、ログ配置ドキュメントをESConv主実験に合わせて更新。
- 主要ESConvスクリプトの既定RUN_TAGを発表用既存成果物に合わせ、`reminiscence` 名がESConv実験であることをコメントで明記。
- 旧回想法/legacyスクリプトに非推奨表示を追加。
- チャットアプリの既定LoRAをESConv Bayes-DPO adapterに更新。
- `__pycache__/` と `.pytest_cache/` のみ削除。
