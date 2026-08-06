# Gold-only DPO 500件比較実験

## 目的

ESConv、MathDial、MediTODの目的小コーパスだけを用いた比較baselineです。BASiS学習時に実際に加えたgold preference 500件をそのまま用い、WildChat由来データを含めずにDPO LoRA学習します。

これは生コーパス応答だけを用いるSFTではありません。目的コーパス応答を`chosen`、同一contextに対して既存工程で合成した応答を`rejected`とするGold-only DPOです。

## Stage

```text
prepare_data
→ train
→ generate_responses
→ oracle_eval
→ statistics
→ report
```

- `prepare_data`: gold 500件のtrain由来、重複、BASiS armへの包含、評価会話とのリークを監査し、バイト同一コピーを作ります。
- `train`: 既存BASiSと同じQwen/LoRA/DPO条件で1 epoch学習します。500件のため約63 optimizer stepです。
- `generate_responses`: 既存評価成果物に保存された`model_prompt`を変更せず、Gold-only adapterの応答だけを生成します。
- `oracle_eval`: 既存と同じrubric、10段階、judge modelでGold-onlyだけを採点します。既存3モデルrawは読み取り専用です。
- `statistics`: 4群Friedman、Kendall's W、有意時の6ペアHolm補正、効果量、bootstrap 95% CIを計算します。
- `report`: 学習条件、hash、軸別平均、統計結果、解釈上の制約をMarkdownへまとめます。

## 一括実行

```bash
RUN_TAG=gold_only_dpo500_v1 \
WORKERS=4 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
EVAL_CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_DEVICE_MAP=auto \
TRAIN_MAX_MEMORY='0=46GiB,1=46GiB,cpu=0GiB' \
PYTHONUNBUFFERED=1 \
./scripts/run_gold_only_dpo_all_watchdog.sh
```

`ESConv → MathDial → MediTOD`の順に逐次実行します。GPUへ複数モデルを同時に載せません。

## Dataset・stage単位の再開

```bash
DATASET=mathdial \
RUN_TAG=gold_only_dpo500_v1 \
START_STAGE=generate_responses \
END_STAGE=report \
WORKERS=4 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
EVAL_CUDA_VISIBLE_DEVICES=0,1 \
./scripts/run_gold_only_dpo_dataset_watchdog.sh
```

同じ`RUN_TAG`では、入力、config、既存Oracle raw、関連コードのhashが一致する場合だけSUCCESS markerを再利用します。API採点と応答生成は1件ごとに保存され、学習は`resume-from-checkpoint auto`で再開します。

## 出力

```text
artifacts/gold_only_dpo/runs/<RUN_TAG>/<dataset>/
  data/gold_only_train.jsonl
  data/gold_only_manifest.json
  training/gold_only_lora/
  evaluation/gold_only_responses.jsonl
  evaluation/oracle_gold/
  evaluation/oracle_combined/
  statistics/
  statistics_ood/          # MediTODのみ
  reports/final_report.md
  stage_markers/
  logs/
```

MathDialは既存のoutcome-selected 100件を用いる探索的比較です。MediTODはin-domainを主評価、OOD 30件を副次評価とし、診療単位のcluster-aware感度分析も出力します。既存3モデルとGold-onlyのOracle採点時期が異なる点はレポートに制約として残します。
