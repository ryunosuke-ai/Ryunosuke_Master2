# MathDial × WildChat-1M BASiS pipeline

各stageの処理、入出力、使用モデル、成功判定、監視方法の詳細は
`docs/MATHDIAL_WILDCHAT_STAGE_GUIDE.md`を参照する。

## 目的とESConv再利用

本実験は、ESConv × DailyDialogで用いたBASiSの処理を、MathDial × WildChat-1Mへ
置き換えて再現性を確認する追加実験である。既存ESConvコードと成果物形式は変更しない。

再利用する中心処理:

- `core/transition_bayes_model.py`: ベイズモデル検証とposterior更新
- `tools/score_dialogue_with_transition_bayes_model.py`: 大コーパスLLM観測評価
- `tools/extract_high_posterior_dialogues.py`: posteriorと会話別上限による選別
- `tools/translate_and_generate_dpo.py`: 英日翻訳、同一context rejected生成、再スコア
- `tools/build_random_dailydialog_dpo.py`: Random-DPO一般品質生成
- `tools/train_qwen35_dpo_lora.py`: Qwen DPO/LoRA
- `core/oracle_eval_common.py`: 10段階Oracle、retry、resume、集計
- `scripts/analyze_oracle_three_model_significance.py`: Friedman、permutation、Holmの数値処理

MathDial固有部分はadapter、ontology、prompt、評価軸、候補filterとして追加する。

## LLMの役割分担

- 代表会話分析と遷移ベイズモデル直接生成: `MATHDIAL_ANALYSIS_LLM_MODEL`（既定は`AZURE_OPENAI_GPT56_SOL_DEPLOYMENT`、`gpt-5.6-sol`）
- WildChat観測ラベル評価とBASiSスコアリング: `MATHDIAL_SCORING_LLM_MODEL`（既定は`AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT`、`gpt-5.6-terra`）
- 日本語翻訳と同一contextのrejected生成: `MATHDIAL_DPO_GENERATION_MODEL`（既定はTerra）
- 評価promptの日本語化: Terra
- Oracle評価: `MATHDIAL_JUDGE_MODEL`（既定はTerra）
- 学習・評価応答生成: `LOCAL_QWEN_MODEL_ID`（既定 `Qwen/Qwen3.5-27B`）

モデル名はすべて環境変数で上書きでき、実際に使用した値は
`run_metadata.json`へ保存する。

パイプラインshはリポジトリ直下の`.env`を最初に読み込む。GPT-5.6 Azure接続では
`AZURE_OPENAI_GPT56_ENDPOINT`、`AZURE_OPENAI_GPT56_API_VERSION`、
`AZURE_OPENAI_GPT56_API_KEY`を共通で使い、Sol/Terraのdeploymentだけを処理別に切り替える。

## 比較条件

- MathDial-BASiS-DPO: WildChat `basis_top` 2,000件 + MathDial train gold 500件
- MathDial-Random-DPO: WildChat `domain_random` 2,500件、goldなし
- 両armの総学習件数は2,500件
- ベースモデル、翻訳・生成モデル、学習hyperparameterは固定
- 特徴抽出・WildChat選別は英語、DPO学習と最終評価は日本語

## Stage

`preprocess`, `build_basis`,
`extract_wildchat`, `score_wildchat`, `select_data`, `build_dpo`, `train`,
`generate_responses`, `oracle_eval`, `statistics`, `report`の順に実行する。

```bash
# API/GPUを使わない全stage dry-run
./scripts/run_mathdial_wildchat_dry_run.sh

# 本実験
RUN_TAG=mathdial_wildchat_gpt56_v2 ./scripts/run_mathdial_wildchat_watchdog.sh

# 区間実行
RUN_TAG=mathdial_wildchat_gpt56_v2 \
START_STAGE=score_wildchat END_STAGE=build_dpo \
./scripts/run_mathdial_wildchat_watchdog.sh

# 1 stageだけ実行
RUN_TAG=mathdial_wildchat_gpt56_v2 STAGE=statistics \
./scripts/run_mathdial_wildchat_watchdog.sh
```

完了stageは`stage_state/*_SUCCESS.json`でskipする。再実行には
`FORCE_STAGE=stage_name`を指定する。`LIMIT`、`WORKERS`、`SEED`、各モデル環境変数、
`TRAIN_CUDA_VISIBLE_DEVICES`を変更できる。

## 本実行前の確認

1. `build_basis`の80会話がtrain-only、qid一意、4 Teacher moveを網羅しているか確認する。
2. WildChat候補数がBASiS 2,000件、Random 2,500件を十分上回るか確認する。
3. 翻訳errorで数値・数式が失われていないか確認する。
4. BASiS armだけがMathDial gold 500件を含み、Random armがgold 0件か監査する。
5. 評価prompt manifestを応答生成前に固定する。
6. Oracle結果を見る前に`mathdial_oracle_v1`の主5軸を変更しない。

WildChat全走査、LLM本抽出、DPO API生成、GPU学習、Oracle本評価はdry-runでは実行されない。
