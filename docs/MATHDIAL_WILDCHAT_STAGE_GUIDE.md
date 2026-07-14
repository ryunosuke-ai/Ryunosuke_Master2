# MathDial × WildChat-1M stage guide

## 1. 目的

このパイプラインは、ESConv × DailyDialogで用いたBASiSの役割を変えず、対象小コーパスをMathDial、大規模候補をWildChat-1Mへ置き換える。

```text
MathDial train代表会話
  -> Solによる会話特徴分析と遷移ベイズモデル生成
  -> TerraによるWildChat観測ラベル判定
  -> posterior・遷移・多様性によるBASiS選別
  -> 英語文脈とchosenを日本語化し、同一文脈のrejectedをLLM生成
  -> Qwen DPO + LoRA
  -> held-out MathDial履歴で3モデル応答生成
  -> blind Oracle評価
  -> 対応あり統計検定
```

旧方式の発話別`extract_features`と`validate_extraction`は主要経路から外した。既存の`features/extractions*.jsonl`は削除しないが、新しい`build_basis`以降では読まない。

## 2. stage一覧

実行順は次の11 stageで固定する。

```text
preprocess
build_basis
extract_wildchat
score_wildchat
select_data
build_dpo
train
generate_responses
oracle_eval
statistics
report
```

### 2.1 `preprocess`

- 公式MathDialを既存adapterで正規化する。
- Studentを`user`、Teacherを`assistant`へ変換する。
- Teacher moveの接頭辞を本文から除き、metadataに保持する。
- 連続Teacher発話、完全一致重複、空発話を既存規則で処理する。
- qid単位でtrain/validation/testを分離し、official testと重なるtrainをquarantineする。
- 完全履歴、response、`next_user_turn`を持つassistant sampleを作る。

主出力:

- `mathdial/data/mathdial_conversations.jsonl`
- `mathdial/data/mathdial_assistant_samples.jsonl`
- `mathdial/reports/preprocessing_statistics.json`

### 2.2 `build_basis`

- trainだけからqid一意な80会話をseed固定で層化抽出する。
- `probing / focus / telling / generic`を全て含むようにする。
- 完全会話、問題、正解、Teacher move annotationをSolへ渡す。
- Solは4〜7 states、4〜8 observationsを持つ`transition_bayes_network`を直接生成する。
- JSON構文が壊れた場合は修復を1回行う。
- schema、確率範囲、各確率行の合計、positive/negative stateを検証する。
- 各正負stateに反対群よりemissionが0.10以上高い識別観測を要求する。
- negative優勢観測を2種類以上要求し、早すぎる解答と文脈不一致を分離する。
- 不合格候補は`rejected_models.jsonl`へ隔離し、canonical modelにはしない。
- 80会話を途中で切らない。入力文字数上限を超えた場合は停止する。

主出力:

- `basis_model/mathdial_analysis_corpus.jsonl`
- `basis_model/mathdial_analysis_corpus.manifest.json`
- `basis_model/mathdial_analysis_prompt.txt`
- `basis_model/mathdial_analysis_input.txt`
- `basis_model/mathdial_transition_bayes_model.json`
- `basis_model/mathdial_transition_compat.json`
- `basis_model/mathdial_transition_bayes_model.manifest.json`
- `basis_model/mathdial_model_quality.json`
- `basis_model/rejected_models.jsonl`（不合格候補があった場合）

### 2.3 `extract_wildchat`

- Hugging Face streamingでWildChat-1Mを走査する。
- 全データを保存せず、英語の教育・個別指導候補だけを保存する。
- 既定で3 user-assistant exchange以上を要求する。
- toxic、redacted、空発話、role破損、完全重複、近似重複を除く。
- general tutoringを主集合、math-onlyをablation集合にする。
- 足場かけの良さなど目的スタイル自体は粗抽出条件にしない。
- checkpointにstream位置、候補、統計を保存して中断後に再開する。

### 2.4 `score_wildchat`

- Terraで各応答を生成ベイズモデルのobservationへ分類する。
- MathDial presetではstate名やdataset hypothesisをTerraへ見せず、observationだけを分類候補にする。
- 未知ラベルやJSON不正は、許可observationだけを示す短いpromptで最大2回再判定する。
- 最初の200応答をpilotとし、fallback率・不正出力率が各1%以下、かつ有効観測2種類以上であることを確認する。
- pilotは200件を初めて超える会話を丸ごと含めるため、実件数は200件以上になる。会話境界維持により199件で停止することはない。
- WildChat全体の粗候補を、assistant応答を見ずにuser側の混乱・試行・再質問・履歴長で優先順位付けする。
- pilot通過後は20,000応答ずつresume scoringし、各batch後に実際のBASiS選別可能件数を再集計する。
- `SELECTION_POOL_COUNT`（既定5,000件）へ到達した時点で、未評価候補を残してscoringを終了する。
- 各batchのスコア済み件数、選別可能件数、fallback件数は`scoring/selection_pool_history.jsonl`へ追記する。
- conversation内はturn順を維持してposteriorを逐次更新する。
- conversation単位で並列化し、結果を逐次JSONLへ追記する。
- 既存結果のconversation/turn keyはresume時にskipする。
- API/JSON失敗はSDK再試行後にnegative寄りfallbackとして記録する。
- pilotのfallback率が1%を超えた場合は、prompt/API接続異常として本スコアリングへ進まない。
- 本スコアリング完了後は、429・timeout等を含む会話を低並列で丸ごと再評価し、posterior系列も作り直す。
- 本スコアリングのfallback率は1%超で警告、5%超で停止する。警告時も理由別件数を`scoring/fallback_diagnostics.json`へ保存する。

### 2.5 `select_data`

- `domain_random`、`topic_similarity_top`、`basis_top`を同条件で作る。
- positive/negative statesとemission差から優先・除外observationを自動導出する。
- MathDialでは各観測の`max positive emission - max negative emission`を使い、±0.05で優先・中立・除外に分ける。
- 生成モデル固有のラベル名へ依存しない。
- posterior、観測、文脈長、会話単位上限、MMRを既存ESConv選別器で扱う。

### 2.6 `build_dpo`

- BASiS: WildChat選別2,000件 + MathDial train gold 500件。
- Random: WildChat domain random 2,500件、goldなし。
- WildChatの英語履歴とchosenを日本語化する。
- chosenと同じ完全履歴に対する低品質応答候補をTerraで生成し、再スコアする。
- prompt hash一致を必須にし、異なるcontextの応答は組にしない。
- accepted/skippedを逐次保存し、目標件数まで残候補を処理する。
- Random側も同じ生成モデル、候補数、温度条件を使い、逐次保存・resumeする。

### 2.7 `train`

- `Qwen/Qwen3.5-27B`へDPO + LoRAを適用する。
- BASiSとRandomでベースモデル、総件数、hyperparameterを揃える。
- `save_steps=25`でcheckpointを保存する。
- 再開時は`--resume-from-checkpoint auto`を使う。
- OOM時にbatch sizeや学習条件を自動変更しない。

### 2.8 `generate_responses`

- held-out MathDial testからqid一意な約100 promptを作る。
- 問題、誤答、会話履歴を訂正せず日本語へ翻訳する。
- 同じpromptへBase、BASiS-DPO、Random-DPOの応答を生成する。
- prompt単位で逐次保存し、中断後は成功済みpromptをskipする。

### 2.9 `oracle_eval`

- モデル名を見せず、応答順をseed固定でランダム化する。
- MathDial個別指導スタイルと一般品質を別promptで10段階評価する。
- raw、errors、model、prompt version、短い根拠を保存する。
- 成功済みprompt/modelはresume時にskipする。

### 2.10 `statistics`

- 同じpromptの3モデルを対応あり比較する。
- Friedman検定、有意な軸だけの事後比較、Holm補正を行う。
- Kendall's W、効果量、bootstrap 95% CIを保存する。

### 2.11 `report`

- 各manifest、選別診断、評価、統計結果を1つのMarkdownへ集約する。

## 3. モデル割当

`.env`では次を設定する。

```env
AZURE_OPENAI_GPT56_SOL_DEPLOYMENT=gpt-5.6-sol
AZURE_OPENAI_GPT56_TERRA_DEPLOYMENT=gpt-5.6-terra
AZURE_OPENAI_GPT56_API_KEY=...
```

- Sol: `build_basis`の代表会話分析とベイズモデル生成。
- Terra: WildChat scoring、DPO生成、日本語評価prompt翻訳、Oracle評価。
- Local Qwen: DPO学習と3モデル応答生成。

## 4. SUCCESS markerと再開

stage成功時だけ`stage_state/<stage>_SUCCESS.json`を書く。markerには実験fingerprint、入力hash、入力件数、config hash、モデル名を含む。同じ`RUN_TAG`で条件や入力が変わった場合は古いmarkerを再利用しない。

`pipeline_status.json`にはstage、watchdog attempt、主要成功件数、fallback件数・率、`success / incomplete / fatal`を記録する。

一時エラーは15秒、30秒、60秒待って最大3回再試行する。研究結果を無効にする次の問題は致命扱いにする。

- ベイズモデルschema・確率が不正
- train/test/qidリーク
- WildChat候補またはDPO件数不足
- scoring pilot fallback率が1%超、または修復後の本scoring fallback率が5%超
- BASiS 2,000 + gold 500、Random 2,500の構成不一致
- 3モデル評価応答またはOracle必要件数の不足が再試行でも解消しない

## 5. 実行方法

API/GPUなしの確認:

```bash
./scripts/run_mathdial_wildchat_dry_run.sh
```

無人本実行:

```bash
RUN_TAG=mathdial_wildchat_gpt56_v3 \
REUSE_DATA_RUN_TAG=mathdial_wildchat_gpt56_v2 \
WORKERS=8 \
PYTHONUNBUFFERED=1 \
./scripts/run_mathdial_wildchat_watchdog.sh
```

watchdogは30秒ごとに進捗を確認し、既定300秒停止した再開可能stageをprocess group単位でTERM、10秒後も残ればKILLする。再起動後のworkerは4、最大20回である。`build_basis`、学習、統計など行数が増えないstageは単純なstall判定から除外する。

段階実行:

```bash
RUN_TAG=mathdial_wildchat_gpt56_v3 \
REUSE_DATA_RUN_TAG=mathdial_wildchat_gpt56_v2 \
START_STAGE=preprocess \
END_STAGE=build_dpo \
WORKERS=8 \
./scripts/run_mathdial_wildchat_watchdog.sh
```

合格済みv3 basisを維持して、pilot境界修正版へ安全に継続する場合:

```bash
RUN_TAG=mathdial_wildchat_gpt56_v3_resume1 \
REUSE_DATA_RUN_TAG=mathdial_wildchat_gpt56_v3 \
REUSE_BASIS_RUN_TAG=mathdial_wildchat_gpt56_v3 \
START_STAGE=preprocess \
END_STAGE=build_dpo \
WORKERS=8 \
PYTHONUNBUFFERED=1 \
./scripts/run_mathdial_wildchat_watchdog.sh
```

この経路は前処理、品質合格済みbasis、WildChat候補だけをhash検証して再利用する。旧scoring、selection、DPO、SUCCESS markerはコピーしない。

20,000件scoring完了後にfallback gateで停止したrunを、429修復から継続する場合:

```bash
RUN_TAG=mathdial_wildchat_gpt56_v3_resume2 \
REUSE_DATA_RUN_TAG=mathdial_wildchat_gpt56_v3 \
REUSE_BASIS_RUN_TAG=mathdial_wildchat_gpt56_v3 \
REUSE_SCORING_RUN_TAG=mathdial_wildchat_gpt56_v3_resume1 \
START_STAGE=preprocess \
END_STAGE=build_dpo \
SELECTION_POOL_COUNT=5000 \
SCORING_BATCH_RECORDS=20000 \
WORKERS=8 \
SCORING_REPAIR_WORKERS=4 \
PYTHONUNBUFFERED=1 \
./scripts/run_mathdial_wildchat_watchdog.sh
```

この経路はraw scoring 20,000件を照合して再利用する。既存WildChat checkpointから残りを全走査し、粗候補をuser側の指導機会でグローバルに並べる。
以降は20,000件ずつ追加評価し、選別可能候補5,000件へ達した時点で自動停止する。429・timeoutを含む会話は低並列で再評価する。selectionとDPOは再利用しない。

実行途中の同じrunで、保存済みscoringを維持したまま判定batchだけを3,000件へ縮小する場合は、
`scripts/resume_mathdial_scoring_small_batches.sh`をwatchdogのpipelineとして指定する。
元runのfingerprintは変更せず、変更理由、開始件数、batch幅を
`scoring/scoring_configuration_amendments.jsonl`へ保存する。目標到達後は元のfingerprint条件で
`select_data`から再開する。

単一stageの明示再実行:

```bash
RUN_TAG=mathdial_wildchat_gpt56_v3 \
STAGE=score_wildchat \
FORCE_STAGE=score_wildchat \
./scripts/run_mathdial_wildchat_watchdog.sh
```

## 6. 停止

watchdogを前面実行している端末では`Ctrl+C`でwatchdogを止める。watchdogは子pipelineのprocess group全体へTERMを送り、必要ならKILLする。tmux外から特定runだけ停止する場合は次を使う。

```bash
kill -TERM "$(cat artifacts/mathdial_wildchat/runs/mathdial_wildchat_gpt56_v3/watchdog/watchdog.pid)"
```

watchdog script名や広い`python`を条件にした`pkill`は、他の実験を巻き込むため使わない。
