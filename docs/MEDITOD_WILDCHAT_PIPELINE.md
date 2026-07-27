# MediTOD × WildChat BASiS実験

## 位置づけ

この実装は、ESConv/MathDialで用いたBASiSの中心処理を変更せず、目的小コーパスをMediTODへ置き換える。変更箇所はMediTOD adapter、病歴聴取用分析prompt、WildChat健康相談filter、scoring preset、評価軸だけである。

```text
MediTOD train
  → Solによる小コーパス分析と遷移ベイズモデル生成
  → WildChat健康相談のstreaming抽出
  → Terraによる観測分類と既存posterior更新
  → MMRを含むBASiS選別
  → 日本語DPO (BASiS 3,000 + gold 500 / Random 3,500)
  → Qwen3.5-27B DPO LoRA
  → Base / BASiS / Randomのblind Oracle評価・統計
```

## データ

- MediTOD revision: `0b7a87c3553c9056bc8371b0809811003f94b261`
- WildChat-1M revision: `7d6490e462285cf85d91eabea0f9a954fbddcd1f`
- public raw: 本文hashで231 recordを213診療へ統合し、`meditod_public_raw_v1`としてtrain 160 / validation 20 / test 18 / OOD 15へ分ける。この分割を公式splitとは呼ばない。
- canonical full: UMLS利用条件と公式申請を満たしたデータだけを`MEDITOD_SOURCE_MODE=canonical_full`で別runへ入力する。public raw成果物と混ぜない。
- `control`発話は本文から除外し、annotationとして保存する。doctorはassistant、patientはuserへ変換する。

実データの軽量監査では、18重複組、split間本文hash leakage 0、train 24代表診療2,337発話を確認した。

## Stage

| stage | 処理 |
|---|---|
| `preprocess` | MediTOD正規化、重複統合、split、assistant sample、統計・leakage監査 |
| `build_basis` | train 24完全診療と全train annotation集計をSolへ渡し、遷移ベイズモデルを直接生成 |
| `extract_wildchat` | WildChatをstreaming走査し、一般健康相談と呼吸器ablationを保存 |
| `score_wildchat` | Terra分類、pilot、3,000件単位のadaptive scoring、clean pool測定 |
| `select_data` | `basis_top / domain_random / topic_similarity_top`、state-specific margin、MMR |
| `build_dpo` | 医療情報を保った日本語化、同一context rejected、gold追加、件数監査。BASiS/Random不足時は既存poolへ3,000件単位で追加scoring |
| `train` | Base共通のQwen3.5-27BへBASiS/Randomを同条件でDPO LoRA学習 |
| `prepare_eval` | in-domain 100件とOOD補助標本を事前層化し、日本語化。空翻訳・発話数・role順を必須検査し、医療情報差分は採否に使わず監査warningへ保存 |
| `generate_responses` | 同一promptへBase/BASiS/Random応答を生成 |
| `oracle_eval` | 病歴聴取7軸、一般品質5軸、安全性proxy 5軸をblind評価 |
| `statistics` | Friedman、Holm事後比較、効果量、bootstrap CI、診療cluster感度分析 |
| `report` | run内の統計と監査情報をMarkdown化 |
| `prepare_user_eval` | Oracle上のBASiS優位項目から副次的人手評価A/Bを作成 |

`score_wildchat / build_dpo`は固定clean pool上限を持たない。主候補は
`wildchat_health_broad.v4`の広域健康ドメイン集合であり、4 user turns以上、
assistant後の追加情報、健康関連性を粗条件とする。本人の症状相談、論文要約、
文章作成、教育用途などの用途分類は主実験の採否に使わない。個人健康相談判定は
診断用ablationとして集計するだけで、BASiSが状態・戦略・遷移に基づいて
目的スタイルを選別する。

未処理候補を3,000件単位でscoring・再選別し、全候補を使い切った場合だけ、
安全性、同一context、`chosen > rejected`、score gap 0.10以上を満たす
厳格基準未達ペアを順位救済する。救済件数と条件はmanifestへ保存する。
有限のsourceをすべて使ってもこの条件を満たす3,000件が存在しない場合は、
件数を水増しせず品質エラーとして停止する。

## 実行

`.env`へ次を設定する。

```env
AZURE_OPENAI_GPT56_SOL_DEPLOYMENT=gpt-5.6-sol
AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT=gpt-5.6-terra
AZURE_OPENAI_GPT56_API_KEY=...
LOCAL_QWEN_MODEL_ID=Qwen/Qwen3.5-27B
```

API/GPUを使わない確認:

```bash
RUN_TAG=meditod_wildchat_dry_run \
./scripts/run_meditod_wildchat_dry_run.sh
```

本実験は、十分なディスクとGPU空きを確認してwatchdogから実行する。
公式リポジトリ（canonical完全版ではUMLSを含む）の利用条件を確認したうえで、
確認記録用の`MEDITOD_DATA_TERMS_CONFIRMED=1`を指定する。

```bash
RUN_TAG=meditod_wildchat_gpt56_v1 \
MEDITOD_DATA_TERMS_CONFIRMED=1 \
START_STAGE=preprocess \
END_STAGE=prepare_user_eval \
WORKERS=4 \
SCORING_REQUESTS_PER_MINUTE=120 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
EVAL_CUDA_VISIBLE_DEVICES=0,1 \
PYTHONUNBUFFERED=1 \
./scripts/run_meditod_wildchat_watchdog.sh
```

stage単独または範囲再開は、同じ`RUN_TAG`と同じ実験条件を維持して`START_STAGE / END_STAGE`を指定する。config、モデル、件数等が変わる場合は新しい`RUN_TAG`を使う。

旧v2の全体走査、scoring、採択済みDPOを品質監査して再利用する場合は、
次の互換移行名を明示する。全体走査済みartifactはdataset revision、seed、
件数、file hashを検証し、`reuse_manifest.json`へ保存する。strict personal
filterで隔離された採択結果は、source/prompt/model/Bayes/threshold/contextと
医療情報保持検査v3を通過したものだけ復元する。元quarantineは監査履歴として
変更しない。旧fidelity検査だけで失敗したサンプルは再処理する。

```bash
RUN_TAG=meditod_wildchat_gpt56_v2 \
MEDITOD_DATA_TERMS_CONFIRMED=1 \
MEDITOD_RESUME_MIGRATION=target3000_broad_health_fidelity_v3 \
START_STAGE=score_wildchat \
END_STAGE=prepare_user_eval \
WORKERS=4 \
SCORING_REQUESTS_PER_MINUTE=120 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
EVAL_CUDA_VISIBLE_DEVICES=0,1 \
PYTHONUNBUFFERED=1 \
./scripts/run_meditod_wildchat_watchdog.sh
```

医療情報保持検査v3は、角括弧内の論文引用番号と非臨床文書の識別番号を
保持対象から外す。一方、個人相談の数値、投薬量、期間、年齢、バイタル、
検査値などは保持し、不一致時はprompt/chosen翻訳だけを最大2回修復する。
rejected候補は修復時に再生成しない。

評価promptの日本語化では、fidelity判定による標本の偏りを避けるため、
数値・薬剤・症状等の差分をhard gateにしない。差分は
`translation_fidelity_warnings.jsonl`と選定manifestへ保存し、原文と訳文は
評価prompt JSONLに保持する。空翻訳、発話数不一致、role順の破損だけを
評価不能として除外する。v2の学習済みadapterから評価だけを再開する場合:

```bash
RUN_TAG=meditod_wildchat_gpt56_v2 \
MEDITOD_DATA_TERMS_CONFIRMED=1 \
MEDITOD_RESUME_MIGRATION=eval_fidelity_audit_only_v6 \
MEDITOD_BASIS_SELECTED_COUNT=1824 \
MEDITOD_GOLD_COUNT=500 \
MEDITOD_RANDOM_COUNT=2324 \
START_STAGE=prepare_eval \
END_STAGE=prepare_user_eval \
WORKERS=4 \
SCORING_REQUESTS_PER_MINUTE=120 \
TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
EVAL_CUDA_VISIBLE_DEVICES=0,1 \
PYTHONUNBUFFERED=1 \
./scripts/run_meditod_wildchat_watchdog.sh
```

## 人手評価

MediTOD本run完了後:

```bash
FORM_ROOT=artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v2_posthoc_axes \
DATABASE=artifacts/user_eval/web/meditod_likert_responses_v2.sqlite3 \
PUBLIC_HOST=192.168.1.17 \
PORT=8505 \
./scripts/run_meditod_likert_user_eval_web.sh
```

MathDial事後選択100件から人手評価itemを作成し、別DBで起動する:

```bash
./scripts/prepare_mathdial_likert_user_eval.sh

FORM_ROOT=artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/user_eval_v2_posthoc_axes \
DATABASE=artifacts/user_eval/web/mathdial_likert_responses_v2.sqlite3 \
PUBLIC_HOST=192.168.1.17 \
PORT=8504 \
./scripts/run_mathdial_likert_user_eval_web.sh
```

公開itemはモデル名、Oracle値、正解位置を含まない。復号情報は`private_answer_key.jsonl`だけに保存する。選定には事後採用軸、BASiS優位幅、軸別勝数、応答間の文章差、人手可読性監査を用いる。標本は`outcome_enriched_secondary_human_eval_posthoc_axes`であり、test全体の無条件な主結果として解釈しない。

回答統計:

```bash
python3 -m tools.analyze_three_model_likert_responses \
  --database artifacts/user_eval/web/meditod_likert_responses_v2.sqlite3 \
  --definition configs/user_evaluations/meditod_likert_v2.yaml \
  --private-answer-key artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v2_posthoc_axes/private_answer_key.jsonl \
  --output-dir artifacts/user_eval/web/meditod_statistics
```

氏名を含むDBとCSVは個人情報であり、commit/pushしない。

## 評価上の注意

評価軸の根拠は[評価軸対応表](MEDITOD_EVALUATION_AXES.md)に固定した。Oracleの安全性カテゴリは臨床専門家評価ではなく補助proxyであり、医学的安全性を保証しない。1診療から複数promptを取るため、prompt単位の主分析に加えて診療単位へ集約した感度分析を必ず報告する。
