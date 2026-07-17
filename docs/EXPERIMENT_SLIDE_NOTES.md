# 中間発表用: ESConv/BASiS DPO実験整理

作成日: 2026-06-18
根拠: このリポジトリ内のコード、ログ、生成済みデータ、評価summary/manifestを確認した。

注意: `BASiS` という名称はリポジトリ内では直接出現しない。本資料では、BASiSを「Bayesian dialogue modelのposteriorに基づくBayes-selectedデータ選別」として整理する。成果物名に `reminiscence_5000_to_2000` が残っているが、`docs/results/README.md` にある通り、発表用の主実験の実体はESConv支援対話スタイル学習実験である。

## 1. 実験設定の要約

### 目的

本実験は、小規模高品質コーパスであるESConvからLLMで会話状態・応答戦略・状態遷移を分析し、その分析結果から生成したBayesian dialogue modelを使って、大規模一般対話コーパスDailyDialogからESConvらしい応答を選別できるかを検証する。

検証したい効果は2つに分かれる。

| 検証したい効果 | 比較 | 説明 |
|---|---|---|
| DPOそのものの効果 | Base Qwen vs Bayes-DPO | Qwen/Qwen3.5-27BにDPOを行うことで、ESConvらしい支援応答が増えるかを見る。 |
| Bayesian dialogue modelによるデータ選別の効果 | Bayes-DPO vs Random-DPO | どちらもDPO済みだが、Bayes-DPOはposterior選別、Random-DPOはランダム抽出なので、DPO一般の効果ではなくBASiS/Bayes選別の寄与を切り分けられる。 |
| Oracle依存性 | gpt-5.4-pro Oracle vs gpt-5.4 Oracle | 評価LLMを変えてもBase vs Bayes-DPOの傾向が残るかを見る。 |

### 全体の処理フロー

```text
ESConv小コーパス 80会話 / 2254発話
  -> gpt-5.4-proで状態遷移Bayes modelを生成
  -> DailyDialog候補をgpt-5.4で観測ラベル評価
  -> transition Bayes modelでposteriorを計算
  -> 高posterior応答を抽出
  -> 日本語DPO候補を生成し、chosen/rejectedを再スコア
  -> DailyDialog由来2000件 + ESConv gold 500件を混合
  -> Qwen/Qwen3.5-27BをDPO LoRA学習
  -> Base / Bayes-DPO / Random-DPOをOracle評価
```

## 2. 使用データ

| データ | 実体 | 役割 | 実際の件数 |
|---|---|---|---:|
| 小規模高品質コーパス | ESConv, `thu-coai/esconv` | Bayesian dialogue model生成とOracle評価時の参照スタイル | 80会話 / 2254発話 |
| 大規模一般対話コーパス | DailyDialog, `ConvLab/dailydialog` train split | Bayes-selected DPO候補およびRandom baseline候補 | chunk集計で53810候補 |
| Bayes-selected英語候補 | DailyDialogをposteriorで選別 | 日本語DPO生成の入力 | chunk集計で6400件抽出 |
| Bayes-DPO DailyDialog | Bayes-selected候補から日本語DPO化 | Bayes-DPO学習データの主成分 | 2000件 |
| ESConv gold DPO | ESConvから作成 | 小規模高品質データを明示的に混合 | 500件 |
| Bayes-DPO mixed | DailyDialog 2000 + ESConv gold 500 | Bayes-DPO LoRAの学習データ | 2500件 |
| Random-DPO | DailyDialogをseed=42でランダム抽出 | Bayes選別なしDPO baseline | 2500件 |
| Oracle評価prompt | ESConv v3 strategy prompt | Base/Bayes/Random比較の評価入力 | 100件、10カテゴリ x 10件 |

補足:

- Bayes-DPO mixedの内訳は `artifacts/datasets/esconv_mixed_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl` の実カウント。
- Random-DPOは `artifacts/datasets/dailydialog_random2500_ja_dpo_preferences_esconv_5000_to_2000_random2500.manifest.json` で `bayes_selection_used=false`, `bayes_model_used=false` と記録されている。
- ESConv goldのstrategy分布は、`Restatement or Paraphrasing` 68、`Question` 68、`Affirmation and Reassurance` 67、`Others` 67、`Providing Suggestions` 66、`Reflection of feelings` 65、`Information` 59、`Self-disclosure` 40。

## 3. データ選別とDPOペア作成

### Bayesian dialogue model

使用モデル: `artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json`

このモデルは、会話状態 `states`、望ましい状態 `positive_states`、望ましくない状態 `negative_states`、観測ラベル `observations`、状態遷移確率 `transition_likelihoods`、観測尤度 `emission_likelihoods` を持つ。

主な状態:

| 種別 | ラベル |
|---|---|
| positive states | `opening_rapport`, `clarify_problem`, `empathic_support`, `collaborative_planning`, `closing_encouragement` |
| negative state | `misaligned_support` |
| observations | `greet_checkin`, `explore_question`, `reflect_validate`, `practical_guidance`, `self_disclosure`, `close_meta`, `misattuned_generic` |

スコアリングでは、LLMが応答を観測ラベルに分類し、その観測ラベルを状態遷移Bayes modelに入れて状態分布を更新する。最終的なstyle-likeness scoreは、positive statesのposterior合計である。

```text
predicted[next_state] = Σ prior[previous_state] * transition[previous_state][next_state]
weighted[state] = predicted[state] * emission[state][observation]
state_posteriors[state] = weighted[state] / Σ weighted
posterior = Σ state_posteriors[state] for state in positive_states
```

### DailyDialog候補の選別

`scripts/run_dpo_pipeline_esconv_2000_chunked.sh` の設定:

| 項目 | 値 |
|---|---:|
| `MAX_DIALOGUES` | 8000 |
| chunk数 | 16 |
| chunkあたり選別数 | 400 |
| `MIN_POSTERIOR` | 0.72 |
| `TARGET_SELECTED_PER_CHUNK` | 400 |
| 選別済み候補合計 | 6400 |

選別処理は `tools/extract_high_posterior_dialogues.py`。`--bayes-model` 指定により、positive/negative statesから優先・除外ラベルを導出し、posteriorだけでなく `selection_score` も付与する。

### 日本語DPOペアの作成

`tools/translate_and_generate_dpo.py` で、選別済み英語候補から日本語DPOペアを作る。

| 項目 | 値 |
|---|---:|
| 生成モデル | `gpt-5.4` |
| スコア再評価モデル | `gpt-5.4` |
| style preset | `esconv_support` |
| rejected候補数 | 8 |
| max output tokens | 6144 |
| strict条件: chosen posterior | `>= 0.70` |
| strict条件: rejected posterior | `<= 0.55` |
| strict条件: score gap | `>= 0.25` |
| gap rescue条件: rejected posterior | `<= 0.65` |
| gap rescue条件: score gap | `>= 0.30` |

作り方:

1. Bayes-selected英語応答を `translated_chosen` として日本語化する。
2. 同じpromptに対する `rejected_candidates` を複数生成する。
3. 日本語のchosen/rejectedを再度Bayes modelでスコアリングする。
4. `chosen_posterior - rejected_posterior` が最大のrejectedを選ぶ。
5. 閾値を満たすペアだけDPO JSONLに採用する。

Bayes-DPO mixed 2500件の集計:

| 指標 | 値 |
|---|---:|
| 件数 | 2500 |
| DailyDialog由来 | 2000 |
| ESConv gold由来 | 500 |
| chosen posterior平均 | 0.9620 |
| rejected posterior平均 | 0.5748 |
| score gap平均 | 0.3871 |
| strict採用 | 506 |
| gap rescue採用 | 1983 |
| acceptance_rule未設定 | 11 |

### Random baseline

`tools/build_random_dailydialog_dpo.py` と `scripts/run_dpo_pipeline_esconv_random_2500.sh` で作成。

| 項目 | 値 |
|---|---:|
| source dataset | DailyDialog |
| source candidates | 53810 |
| target records | 2500 |
| seed | 42 |
| generation model | `gpt-5.4` |
| candidates | 4 |
| style preset | `general_conversation_quality` |
| Bayes selection | なし |
| ESConv gold混合 | なし |

Random-DPOはposteriorやBayesian modelを使わず、DailyDialog候補をseed固定でランダム化し、一般会話品質のchosen/rejectedを生成する。したがって、Bayes-DPO vs Random-DPOは「同じDPOでも、BASiS/Bayes選別を使ったか」の比較になる。

## 4. 学習設定

共通設定:

| 項目 | Bayes-DPO | Random-DPO |
|---|---:|---:|
| base model | `Qwen/Qwen3.5-27B` | `Qwen/Qwen3.5-27B` |
| 学習手法 | DPO + LoRA | DPO + LoRA |
| 学習データ件数 | 2500 | 2500 |
| train/eval | 2500 / 0 | 2500 / 0 |
| epoch | 1 | 1 |
| learning rate | 5e-6 | 5e-6 |
| beta | 0.1 | 0.1 |
| per-device train batch size | 1 | 1 |
| gradient accumulation steps | 8 | 8 |
| LoRA r | 8 | 8 |
| LoRA alpha | 16 | 16 |
| LoRA dropout | 0.05 | 0.05 |
| target modules | `all-linear`相当 | `all-linear`相当 |
| 4bit | 無効 | 無効 |
| device map | auto | auto |
| max memory | `0=46GiB,1=46GiB,cpu=0GiB` | `0=46GiB,1=46GiB,cpu=0GiB` |
| save steps | 25 | 25 |
| warmup ratio | 0.03 | 0.03 |
| seed | 42 | 42 |

学習完了ログ:

| モデル | output | train loss | runtime | step |
|---|---|---:|---:|---:|
| Bayes-DPO | `artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit` | 0.3111 | 6428 sec | 313 |
| Random-DPO | `artifacts/training_runs/qwen35_random2500_dailydialog_dpo_lora_esconv_5000_to_2000_random2500_ep1_lr5e-6_r8_a16_no4bit` | 0.5321 | 6215 sec | 313 |

## 5. 比較対象

| 比較 | 実験目的 | Oracle | 評価件数 |
|---|---|---|---:|
| Base vs Bayes-DPO | DPOによってESConvらしい支援応答が増えるか | `gpt-5.4-pro` | 100 |
| Base vs Bayes-DPO 再評価 | Oracleモデルを変えても傾向が安定するか | `gpt-5.4` | 100 |
| Bayes-DPO vs Random-DPO | DPO一般ではなくBayes/BASiS選別の効果があるか | `gpt-5.4` | 100 |

Bayes-DPO vs Random-DPOでは、実装互換のためsummary上は `base` fieldがBayes-DPO、`dpo` fieldがRandom-DPOを表す。`docs/results/.../summary.json` の `label_note` と `manifest.json` の `base_field_label=bayes_dpo`, `dpo_field_label=random_dpo` を参照。

## 6. 評価方法

Oracle評価は `tools/run_oracle_evaluation.py` と `tools/run_oracle_evaluation_lora_pair.py` で実行されるLLM-as-a-judge評価である。

手順:

1. 評価prompt 100件を読む。
2. Base/Bayes-DPO、またはBayes-DPO/Random-DPOが同じpromptに応答を生成する。
3. Oracle LLMが小コーパス本文抜粋、Bayes modelの状態・観測ラベル、評価promptを参照して理想応答 `oracle_response` を生成する。
4. Oracle LLMが2つの応答を匿名A/Bとして軸別に0-100点で評価する。
5. `weighted_esconv_overall` の差が1.0点未満ならtie、それ以上なら高い方をwinnerとする。

評価設定:

| 項目 | 値 |
|---|---|
| prompt file | `configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl` |
| style preset | `esconv_strategy_v3` |
| prompt template | `oracle_eval.v2` |
| reference template | `oracle_reference_generation.esconv_strategy.v3` |
| judge template | `oracle_score_against_reference.esconv_strategy.v3` |
| local prompt mode | `instruction` |
| max new tokens | 192 |
| temperature | 0.7 |
| top_p | 0.8 |
| repetition penalty | 1.0 |
| seed | 42 |

## 7. 評価指標

| 指標 | 重み/定義 | 発表での説明 |
|---|---|---|
| ESConv core score | `esconv_strategy_adherence` 0.40 + `emotional_reflection_validation` 0.35 + `premature_advice_avoidance` 0.25 | 本研究の主指標。ESConvらしい支援戦略が再現されたかを見る。 |
| weighted ESConv overall | strategy 0.25 + reflection 0.25 + premature advice avoidance 0.20 + supportive tone 0.10 + contextual grounding 0.10 + progression 0.05 + helpfulness 0.05 | ESConvらしさを重視しつつ、支援応答全体の品質も補助的に見る総合指標。 |
| win rate | weighted overallの差が1.0点以上のサンプルで勝った割合。1.0点未満はtie | 100件中どちらの応答がOracleに選ばれたか。 |
| emotional_reflection_validation | 軸別0-100 | 相手の感情を具体的に拾い、つらさや不安を受け止めているか。 |
| premature_advice_avoidance | 軸別0-100 | 感情理解を飛ばして早すぎる助言・断定・一般論へ行っていないか。 |
| esconv_strategy_adherence | 軸別0-100 | ESConv由来の支援戦略、たとえば感情反映、言い換え、確認質問、小さな提案を文脈に合わせて選べているか。 |
| supportive_tone | 軸別0-100 | 温かく、安全で、相談者が話し続けやすいトーンか。 |
| conversational_progression | 軸別0-100 | 確認質問、問題整理、情報提供、次の一歩などによって会話を前に進めているか。v3では主要3軸から意図的に分離されている。 |

## 8. 実験結果

### Base vs Bayes-DPO: gpt-5.4-pro Oracle

| 指標 | Base | Bayes-DPO | Gap |
|---|---:|---:|---:|
| records | 100 | 100 | - |
| ESConv core score | 82.453 | 90.444 | +7.991 |
| weighted ESConv overall | 83.149 | 88.818 | +5.669 |
| win rate | 28% | 68% | +40pt |
| ties | 4% | - | - |

主な軸別改善:

| 軸 | Base | Bayes-DPO | Gap |
|---|---:|---:|---:|
| premature_advice_avoidance | 83.46 | 94.46 | +11.00 |
| emotional_reflection_validation | 81.52 | 89.42 | +7.90 |
| esconv_strategy_adherence | 82.64 | 88.83 | +6.19 |
| supportive_tone | 87.15 | 91.24 | +4.09 |
| contextual_grounding | 85.94 | 89.06 | +3.12 |
| overall_helpfulness | 82.19 | 85.10 | +2.91 |
| conversational_progression | 79.97 | 61.57 | -18.40 |

### Oracleを変えた再評価

| Oracle | records | core gap | overall gap | Bayes-DPO win rate | Base win rate | tie |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-pro` | 100 | +7.991 | +5.669 | 68% | 28% | 4% |
| `gpt-5.4` | 100 | +6.950 | +4.850 | 65% | 29% | 6% |

Oracleを `gpt-5.4-pro` から `gpt-5.4` に変えても、Bayes-DPOがcore/overall/win rateでBaseを上回る傾向は維持された。

### Bayes-DPO vs Random-DPO: gpt-5.4 Oracle

注意: summary上の `Base` はBayes-DPO、`Random-DPO` はsummary上の `DPO` fieldである。

| 指標 | Bayes-DPO | Random-DPO | Gap = Random - Bayes |
|---|---:|---:|---:|
| records | 100 | 100 | - |
| ESConv core score | 90.163 | 83.152 | -7.011 |
| weighted ESConv overall | 88.616 | 83.845 | -4.770 |
| win rate | 62% | 31% | -31pt |
| ties | 7% | - | - |

Bayes-DPOから見た主な軸別差分:

| 軸 | Bayes-DPO | Random-DPO | Bayes - Random |
|---|---:|---:|---:|
| emotional_reflection_validation | 89.39 | 79.67 | +9.72 |
| premature_advice_avoidance | 94.24 | 87.23 | +7.01 |
| supportive_tone | 90.74 | 84.91 | +5.83 |
| esconv_strategy_adherence | 88.29 | 83.65 | +4.64 |
| overall_helpfulness | 84.43 | 82.84 | +1.59 |
| contextual_grounding | 88.37 | 87.22 | +1.15 |
| conversational_progression | 64.30 | 84.29 | -19.99 |

解釈:

- Bayes-DPOはRandom-DPOより、ESConv core score、weighted overall、win rateで上回った。
- 特に感情反映、早すぎる助言の回避、支援的トーンでBayes-DPOが高い。
- 一方、conversational_progressionはRandom-DPOが高い。Random-DPOは一般会話品質や会話を進める応答を生成しやすいが、ESConv的な「まず受け止める」「急いで助言しない」方向は弱い可能性がある。

## 9. 考察に使えるポイント

### 強調できること

- Base vs Bayes-DPOで、Bayes-DPOはESConv core scoreを+7.991、weighted overallを+5.669改善し、win rateも68%だった。
- Oracleを `gpt-5.4` に変えても、core gap +6.950、overall gap +4.850、win rate 65%で、傾向は維持された。
- Bayes-DPO vs Random-DPOでは、Bayes-DPOがcoreで+7.011、overallで+4.770、win rateで62%対31%と上回った。
- 改善の中心は、`emotional_reflection_validation`、`premature_advice_avoidance`、`esconv_strategy_adherence`。これは「ESConvらしさ」を測るcore指標と整合している。
- Random-DPOもDPO済みなので、Bayes-DPOの優位は単なるDPO効果ではなく、Bayesian dialogue modelによるデータ選別の効果を示唆している。

### 注意して説明すべきこと

- `conversational_progression` はBaseやRandom-DPOよりBayes-DPOが低い。これは、Bayes-DPOが感情反映・受容・助言抑制を強く学習し、質問や具体的次ステップの提示を控えめにする方向へ寄った可能性がある。
- `conversational_progression` はweighted overallでは5%の補助軸で、core scoreには含まれない。したがって、主張は「ESConv core戦略は改善したが、会話進行性には課題が残る」が適切。
- Oracle評価はLLM-as-a-judgeであり、人手評価ではない。数値は絶対的な品質ではなく、同一条件下の相対比較として扱う。
- 評価promptは100件であり、より大規模・多様な評価では変動しうる。
- DPOデータ生成でもLLMを使っているため、Bayes modelだけの純粋な効果ではなく、「Bayes選別 + LLM翻訳/negative生成 + 再スコアリング」パイプライン全体の効果として述べるのが安全。
- Random-DPOはBayes選別なし・ESConv gold混合なしなので、Bayes-DPOとの差には「Bayes選別」と「ESConv gold 500件混合」の両方が含まれる。純粋にBayes選別だけを分離するには、Bayes-selected 2500件 vs Random 2500件、かつgold混合なしの追加比較が必要。

推奨表現:

- 「Bayesian dialogue modelによるposterior選別を含むBASiSパイプラインが、ESConv core指標を改善することが示唆された。」
- 「特に感情反映、早すぎる助言の回避、ESConv戦略遵守で改善が見られた。」
- 「一方で、会話を次に進める軸では低下が見られ、受容重視と進行性のバランスが今後の課題である。」

避けた方がよい断定:

- 「BASiSだけが改善要因である」
- 「人間評価でも必ず同じ結果になる」
- 「Bayes-DPOはすべての会話品質でBase/Randomを上回る」
- 「conversational_progressionの低下は必ず悪い」

## 10. スライド構成案

提案された構成で十分。実験パートは次の6枚に分けると説明しやすい。

| スライド | 入れる内容 |
|---|---|
| 実験設定 | 目的、全体フロー、ESConv小コーパス80会話、DailyDialog候補53810、Qwen/Qwen3.5-27B、Bayes-DPO mixed 2500件。 |
| 比較対象と評価 | Base vs Bayes-DPO、Bayes-DPO vs Random-DPO、Oracle変更再評価。DPO効果とBayes選別効果の切り分けを表で示す。 |
| 実験結果1: Base vs Bayes-DPO | core +7.991、overall +5.669、win rate 68%。軸別ではpremature advice avoidance +11.00、reflection +7.90を強調。 |
| 実験結果2: Oracle変更 | gpt-5.4-proとgpt-5.4の比較表。Oracleを変えてもBayes-DPO優位が残ることを示す。 |
| 実験結果3: Bayes-DPO vs Random-DPO | coreでBayesが+7.011、overallで+4.770、win rate 62%対31%。DPO一般ではなくBayes/BASiS選別が効いた可能性を示す。 |
| 考察 | 改善軸、conversational_progression低下の解釈、LLM-as-a-judgeの限界、今後の追加実験。 |

## 11. 参照ファイル

### 結果summary/manifest

- `docs/results/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/summary.json`
- `docs/results/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/manifest.json`
- `docs/results/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/summary.json`
- `docs/results/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy_gpt54/manifest.json`
- `docs/results/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/summary.json`
- `docs/results/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/manifest.json`

### データ・モデル

- `data/esconv_analysis_corpus_reminiscence_5000_to_2000.jsonl`
- `artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json`
- `artifacts/datasets/dailydialog_ja_dpo_preferences_reminiscence_5000_to_2000_daily.jsonl`
- `artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl`
- `artifacts/datasets/esconv_mixed_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl`
- `artifacts/datasets/dailydialog_random2500_ja_dpo_preferences_esconv_5000_to_2000_random2500.jsonl`
- `configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl`

### ログ

- `logs/dpo_pipeline/esconv/20260608/dpo_pipeline_reminiscence_5000_to_2000_20260608_122134.log`
- `logs/dpo_pipeline/random_dpo/20260610/random_dpo_pipeline_esconv_5000_to_2000_random2500_20260610_085013.log`
- `logs/oracle_evaluation/esconv/oracle_eval_v3_strategy_reminiscence_5000_to_2000_20260610_160540.log`
- `logs/oracle_evaluation/esconv/base_vs_bayes_gpt54_20260610_160540.log`
- `logs/oracle_evaluation/bayes_vs_random/oracle_eval_v3_strategy_bayes_vs_random_esconv_5000_to_2000_bayes_vs_random2500_20260610_152413.log`

### スクリプト

- `scripts/run_dpo_pipeline_esconv_2000_chunked.sh`
- `scripts/run_dpo_pipeline_esconv_random_2500.sh`
- `scripts/run_oracle_evaluation_esconv_v3_strategy.sh`
- `scripts/run_oracle_evaluation_esconv_v3_strategy_bayes_vs_random.sh`
- `tools/analyze_esconv_corpus_transition_bayes.py`
- `tools/score_dialogue_with_transition_bayes_model.py`
- `tools/extract_high_posterior_dialogues.py`
- `tools/translate_and_generate_dpo.py`
- `tools/build_esconv_gold_dpo.py`
- `tools/build_random_dailydialog_dpo.py`
- `tools/train_qwen35_dpo_lora.py`
- `tools/run_oracle_evaluation.py`
- `tools/run_oracle_evaluation_lora_pair.py`

## 12. 追加確認が必要な点

- `BASiS` の正式名称・定義はリポジトリ内に見つからなかった。スライドでは研究内で定義してから使う必要がある。
- Bayes-DPO vs Random-DPOは、Bayes選別に加えてESConv gold 500件混合の差も含む。Bayes選別単体の寄与をより厳密に言うなら、gold混合なし条件の追加実験が必要。
- `acceptance_rule` 未設定の11件が混合DPOに残っている。件数は小さいが、発表で厳密性を突かれた場合は、既存runの再開・マージ由来のメタデータ欠落として追加確認する余地がある。
