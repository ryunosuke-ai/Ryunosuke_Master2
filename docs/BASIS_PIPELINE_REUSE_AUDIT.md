# BASiS追加実験 Phase 0 調査結果

## ESConv実験の処理対応

| 処理 | 既存実装 | 再利用方針 |
|---|---|---|
| ESConv読込・正規化 | `tools/prepare_esconv_for_analysis.py` | dataset adapterの設計参考。既存出力は変更しない |
| DailyDialog読込・履歴化 | `tools/prepare_dailydialog_for_scoring.py` | 完全履歴付きsample生成を新共通schemaへ整理 |
| 状態・戦略・遷移抽出 | `tools/analyze_esconv_corpus_transition_bayes.py` | LLM境界とJSON検証を再利用し、promptだけdataset別にする |
| ベイズモデル検証・更新 | `core/transition_bayes_model.py` | 中核アルゴリズムを変更せず共通利用する |
| 大規模コーパス評価 | `tools/score_dialogue_with_transition_bayes_model.py` | retry、resume、逐次事前分布を共通利用する |
| 選別 | `tools/extract_high_posterior_dialogues.py` | posterior、preferred label、会話別上限を再利用する |
| chosen/rejected生成 | `tools/translate_and_generate_dpo.py` | 同一promptへのLLM rejected生成と再スコアを再利用する |
| gold追加 | `tools/build_esconv_gold_dpo.py` | dataset adapterからgold候補を供給する形へ一般化する |
| Random-DPO | `tools/build_random_dailydialog_dpo.py` | seed、件数、生成条件をBASiS群と揃える |
| DPO + LoRA | `tools/train_qwen35_dpo_lora.py` | 学習設定とdry-runを変更せず再利用する |
| 3モデル応答生成・Oracle | `tools/run_oracle_evaluation_lora_pair.py`、`core/oracle_eval_common.py` | blind化、順序乱択、retry/resume、10段階評価を再利用する |
| 有意差検定 | `scripts/analyze_oracle_three_model_significance.py` | Friedman、対応ありpermutation、Holm、Kendall's Wを再利用する |

## 現時点の制約

- ESConvの選別多様性は、層化サンプリング、preferred state/observation、会話別上限まで実装済み。
- MMR、近似重複、strategy/transition/stage coverageの明示的最適化は未実装であり、WildChat選別Gateで追加する。
- 既存ESConvデータ形式と成果物パスは変更せず、新しい共通schemaは追加実験から利用する。
- DPOでは高BASiSスコア応答をchosen候補とし、同じpromptに対するrejected候補をLLMで生成して再スコアする。別prompt間のペアは作らない。

## MathDial事前調査

- 公式Hugging Face revision: `acc3878459e0bd8c04ab840056572f0b8b1abe1f`
- 公式配布件数: train 2,262会話、test 599会話
- 公式train/test間で318 `qid`が重複し、train側585会話が影響する。
- Student発話は`Student:`だけでなくpersona名で記録されるため、`Teacher:`以外の話者をuserへ変換する。
- teacher moveは`focus`、`probing`、`telling`、`generic`の4種だけを本文から除去し、元ラベルをturn metadataへ保持する。
- Hugging Face cardはCC BY 4.0、公式GitHub READMEはCC BY-SA 4.0と記載が異なるため、両方をmanifestへ保存する。
