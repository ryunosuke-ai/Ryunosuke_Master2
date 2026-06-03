# Audit Log

## 2026-06-03 08:30 JST頃

- 対象ファイル:
  - `tools/extract_high_posterior_dialogues.py`
  - `tools/translate_and_generate_dpo.py`
  - `run_dpo_pipeline_reminiscence_2000.sh`
  - `tests/test_dailydialog_pipeline.py`
- 実行した操作:
  - DailyDialogからの候補抽出を、単純な高posterior順ではなく、回想支援型ベイズモデルの状態・観測ラベルを優先する方式へ拡張した。
  - `selection_score` と `selection_reason` を出力し、なぜ採用されたかを追跡できるようにした。
  - 日本語DPO生成に `--target-records` を追加し、accepted件数が目標に達した時点で処理を止められるようにした。
  - 翻訳・rejected生成プロンプトに、過去経験、思い出の情景、当時の気持ち、人間関係、感覚的細部を保持する指示を追加した。
  - 5000会話から候補3200件を抽出し、2000件規模のDPOデータ作成とLoRA/DPO学習を一括実行する `run_dpo_pipeline_reminiscence_2000.sh` を追加した。
- なぜその操作が必要だったか:
  - 250件DPO学習では、ベースモデルとの差が小さく、回想法らしい会話への変化が弱かったため。
  - 単純にデータ数を増やすだけでは、一般的に具体的な雑談が増えるだけで、回想支援に必要な「思い出を深める」信号が薄まる可能性があったため。
  - 研究再現性のため、抽出条件、使用モデル、出力ファイル名、学習設定を1つのスクリプトに固定する必要があったため。
- 代替案があったか:
  - DailyDialog全体からposterior上位だけを採用する案があったが、`warm_closure` や一般的な具体描写が混ざり、回想支援DPOとしての信号が弱くなるため採用しなかった。
  - gpt-5.4-proで全件監査する案があったが、2000件規模では処理時間とAPIコストが大きいため、今回はベイズモデル再スコアリングを主な品質フィルタにした。
  - 既存の `run_dpo_pipeline.sh` を上書きする案があったが、500件実験を再現できるように残し、新規スクリプトを追加した。
- 実行したコマンド:
  - `chmod +x run_dpo_pipeline_reminiscence_2000.sh`
- 変更前後の要約:
  - 変更前: 候補抽出は `posterior` 下限と並び替えだけで、回想支援に有効な状態・観測を明示的に優先できなかった。DPO生成は入力処理件数上限は指定できたが、accepted件数の目標指定はできなかった。
  - 変更後: `opening_invitation`, `setting_sensory_detail`, `activity_social_detail` と `ack_open_probe`, `sensory_setting_focus`, `activity_social_focus` を優先し、`off_style`, `generic_or_unrelated` を除外できる。DPO生成は目標accepted件数で停止できる。
- リスクや注意点:
  - 4000会話のスコアリングと2000件DPO生成はAPI呼び出しが多く、時間と費用が増える。
  - `--require-preferred` により抽出品質は上がるが、候補数が不足する場合がある。その場合は `MAX_DIALOGUES` や `TARGET_SELECTED` を増やすか、`MIN_POSTERIOR` を少し下げる必要がある。
  - DailyDialog由来のため、回想法そのものの応答が十分に多いとは限らない。必要なら次段階で小コーパスを元にした合成DPOデータ追加を検討する。

## 2026-06-03 08:55 JST頃

- 対象ファイル:
  - `tools/audit_dpo_preferences.py`
  - `run_dpo_pipeline_reminiscence_2000.sh`
  - `tests/test_dailydialog_pipeline.py`
- 実行した操作:
  - `gpt-5.4-pro` を使ってDPO preferenceデータを品質監査する `tools.audit_dpo_preferences` を追加した。
  - 監査で合格したDPOサンプルだけを `*_audited.jsonl` として保存し、学習には監査済みデータを使うように一括実行スクリプトを変更した。
  - 監査結果は `*.audit.jsonl` に全件記録し、`quality_score`, `chosen_alignment_score`, `rejected_contrast_score`, `japanese_naturalness_score`, `issues`, `reason` を追跡できるようにした。
- なぜその操作が必要だったか:
  - 今回の研究ではDPOデータの質が研究成果に直結するため、gpt-5.4で大量生成したデータをそのまま学習に使うより、gpt-5.4-proで品質監査して低品質データを除外する方が妥当だったため。
  - 特に、chosenが回想支援として十分に思い出を深めているか、rejectedが自然だが低評価になる比較対象になっているかを追加確認する必要があったため。
- 代替案があったか:
  - 全工程をgpt-5.4-proで実行する案があったが、DailyDialogスコアリング・翻訳・rejected生成の大量処理コストが大きいため採用しなかった。
  - 監査をサンプル抽出だけにする案があったが、学習データの質を直接上げるため、既定ではDPO目標件数分を監査して合格データのみ学習に使う方式にした。
- 実行したコマンド:
  - まだ本番監査APIは実行していない。コード追加とテストのみ。
- 変更前後の要約:
  - 変更前: `--audit-model gpt-5.4-pro` は指定されていたが、通常のDPO生成・学習パイプライン内では品質監査API呼び出しに使われていなかった。
  - 変更後: DPO生成後に `tools.audit_dpo_preferences` が `gpt-5.4-pro` を使って監査し、合格DPO JSONLだけでdry-run確認とLoRA/DPO学習を行う。
- リスクや注意点:
  - `gpt-5.4-pro` 監査によりAPI費用と処理時間が増える。
  - 監査基準が厳しい場合、2000件生成しても監査済み合格データは2000件未満になる可能性がある。その場合は `TARGET_DPO_RECORDS` や `MAX_DIALOGUES` を増やすか、`MIN_AUDIT_QUALITY_SCORE` を調整する必要がある。
  - content filterで監査できないデータは不合格扱いにし、監査レポートに理由を残す。

## 2026-06-03 09:10 JST頃

- 対象ファイル:
  - `run_dpo_pipeline_reminiscence_2000.sh`
- 実行した操作:
  - 一括実行スクリプトから `tools.audit_dpo_preferences` の実行ステップを削除した。
  - 学習データを `*_audited.jsonl` ではなく、DPO生成直後の `dailydialog_ja_dpo_preferences_${RUN_TAG}.jsonl` に戻した。
  - ステップ表記を `8` ステップから `7` ステップへ戻した。
- なぜその操作が必要だったか:
  - 今回はまず「小コーパスの会話に近い会話ができるようになった」という学習結果を得ることを優先するため。
  - 監査プロンプトが回想支援を明示しており、本来の「小コーパスから目的を推定する」研究方針とは少しずれるため。
  - 監査でDPOデータ数が減り、学習効果が弱くなるリスクを避けるため。
- 代替案があったか:
  - 監査プロンプトをベイズモデルの `dataset_hypothesis` に基づく汎用監査へ作り直す案があったが、今回は実験を前に進めるため採用しなかった。
  - 監査をサンプル確認だけに残す案があったが、今回の一括学習パイプラインではデータ数確保を優先し、監査ステップ自体を外した。
- 実行したコマンド:
  - まだ本番パイプラインは実行していない。スクリプト修正のみ。
- 変更前後の要約:
  - 変更前: DPO生成後、`gpt-5.4-pro` 監査を通った `*_audited.jsonl` だけを学習に使う。
  - 変更後: DPO生成後、ベイズ再スコアリングとscore_gap条件を通った通常のDPO JSONLをそのまま学習に使う。
- リスクや注意点:
  - 監査による品質フィルタはなくなるため、一部低品質サンプルが混ざる可能性はある。
  - ただし、データ数は確保しやすくなり、今回の目的である学習効果の確認には向く。

## 2026-06-02 15:10 JST頃

- 対象ファイル:
  - `tools/score_dialogue_with_transition_bayes_model.py`
  - `tests/test_transition_bayes_model.py`
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング再開時に発生したAzure OpenAI `content_filter` エラーへの対応を追加した。
  - content filter発生時、元入力を無視して即除外するのではなく、固有の年齢・日付などを中立プレースホルダへ安全化して1回再評価する処理を追加した。
  - 安全化再試行でもcontent filterにかかった場合のみ、negative/off_style寄り観測へフォールバックし、`llm_error` に理由を記録する処理を維持した。
  - 会話単位の並列スコアリング `--workers` を追加した。
  - `--workers 4` で500会話データの未処理分スコアリングを再開した。
- なぜその操作が必要だったか:
  - DailyDialog内の一部会話がAzure OpenAIのcontent filterに誤検出され、1件で全体処理が停止したため。
  - 研究目的上、問題のない可能性が高いデータを単純に除外せず、できる限り全データを評価する必要があったため。
  - 逐次スコアリングでは残り約2600件の完了に時間がかかりすぎるため。ただし状態遷移の前提を保つため、同一会話内は順序を守り、別会話のみ並列化した。
- 代替案があったか:
  - content filter該当データをすべて低評価扱いにする案があったが、誤検出時に有効データを失うため採用しなかった。
  - content filter該当データを完全にスキップする案があったが、全データ評価方針に合わないため採用しなかった。
  - 並列化せず逐次処理を継続する案があったが、研究作業の時間効率が悪いため採用しなかった。
- 実行したコマンド:
  - `python3 -B -m pytest -p no:cacheprovider -q`
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4`
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filterが発生するとスコアリング処理全体が停止した。スコアリングは逐次処理のみだった。
  - 変更後: content filter発生時は安全化入力で再試行し、成功すれば通常のベイズ更新を行う。再試行も失敗した場合のみnegative寄り観測へフォールバックする。別会話を並列に処理できる。
- リスクや注意点:
  - 安全化入力では具体的な年齢・日付などがプレースホルダ化されるため、細部の意味は一部失われる。ただし今回の評価対象は会話戦略・状態遷移であり、具体値そのものではないため許容した。
  - `--workers 4` によりAPI呼び出しが並列化されるため、レート制限やAPIコストが増える可能性がある。
  - 会話単位で出力へ追記するため、途中停止時には完了済み会話のみ保存される。再実行時は既存出力を読み、保存済みレコードをスキップする。

## 2026-06-02 15:20 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000118#2` と `train_000118#4` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - content filterを完全に無視してAPI評価することはできないが、誤検出の可能性があるため、即スキップや即低評価ではなく、会話戦略評価に必要な構造を保ったまま中立化して評価する必要があった。
- 代替案があったか:
  - 該当レコードをスキップする案があったが、全データ評価方針と品質重視に反するため採用しなかった。
  - 即negativeフォールバックする案があったが、誤検出時に質の高いデータを失う可能性があるため採用しなかった。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filter発生時に処理が停止、または低評価フォールバックのみ。
  - 変更後: content filter発生時に安全化入力で再評価し、成功した場合は通常の観測ラベルとベイズ更新を使用する。
- リスクや注意点:
  - 安全化により具体値の情報は一部失われるが、会話戦略・状態遷移の分類を優先する。
  - 再試行結果には `llm_retry=content_filter_sanitized_retry` が記録されるため、後で該当データを監査できる。

## 2026-06-02 15:30 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000144#11` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - content filterによる停止や安易な除外を避け、可能な限り会話戦略評価を継続するため。
- 代替案があったか:
  - スキップまたはnegativeフォールバックがあったが、品質重視のため安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filterで該当レコード評価が不可能。
  - 変更後: 安全化入力により評価継続を試み、結果には再試行情報を残す。
- リスクや注意点:
  - 安全化で具体情報は一部薄まるため、後続の品質監査で `llm_retry` 付きレコードを確認できるようにする。

## 2026-06-02 15:31 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000144#13` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - 誤検出の可能性があるデータを単純除外せず、会話戦略評価を継続するため。
- 代替案があったか:
  - スキップまたはnegativeフォールバックがあったが、安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filterで該当レコード評価が不可能。
  - 変更後: 安全化入力で再評価を試み、処理全体は継続。
- リスクや注意点:
  - `llm_retry` 付きレコードは、最終DPO採用前に必要に応じて監査対象にできる。

## 2026-06-02 15:45 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000277#6` と `train_000280#3` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - 誤検出の可能性があるレコードを安易に除外せず、全データ評価に近づけるため。
- 代替案があったか:
  - スキップまたはnegativeフォールバックがあったが、品質維持のため安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filter発生時に該当レコードを通常評価できない。
  - 変更後: 安全化入力で再評価を試み、成功すれば通常のベイズ更新に使う。
- リスクや注意点:
  - 安全化再評価済みレコードは `llm_retry` で追跡し、必要なら最終DPO採用前に除外・監査できる。

## 2026-06-02 15:55 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000324#7` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - content filterにより通常入力が評価できなかったが、会話戦略評価を可能な限り継続するため。
- 代替案があったか:
  - スキップまたはnegativeフォールバックがあったが、品質重視のため安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filterで該当レコード評価が不可能。
  - 変更後: 安全化入力で再評価を試み、会話は完了。
- リスクや注意点:
  - `llm_retry` 付きレコードは、必要に応じてDPO採用前の監査対象にする。

## 2026-06-02 15:58 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000332#9` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - content filterの誤検出可能性に対して、安易に除外せず評価継続するため。
- 代替案があったか:
  - スキップまたはnegativeフォールバックがあったが、安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filter発生時に評価が止まる可能性がある。
  - 変更後: 安全化再評価で処理継続を試みる。
- リスクや注意点:
  - 安全化再評価のレコードは追跡可能にし、最終DPO採用時の監査対象にする。

## 2026-06-02 16:08 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000442#4` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - 誤検出の可能性があるcontent filter対象を単純除外せず、会話戦略評価を継続するため。
- 代替案があったか:
  - スキップまたはnegativeフォールバックがあったが、品質維持のため安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filter発生時に該当レコードを評価できない。
  - 変更後: 安全化入力で再評価を試み、処理は継続。
- リスクや注意点:
  - `llm_retry` 付きレコードは後続のDPO採用時に監査可能。

## 2026-06-02 16:09 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000442#5` と `train_000442#7` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - 同一会話内の複数ターンでcontent filterが発生したが、会話全体の状態遷移評価をできる限り保持するため。
- 代替案があったか:
  - 会話全体をスキップする案があったが、全データ評価と品質維持に反するため採用しなかった。
  - 該当ターンだけnegativeフォールバックする案があったが、安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filter対象ターンは通常評価できない。
  - 変更後: 安全化入力で再評価し、可能なら通常の状態遷移更新へ反映する。
- リスクや注意点:
  - content filterが多い会話は最終DPO採用前の監査対象として扱うのが望ましい。

## 2026-06-02 16:18 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
- 実行した操作:
  - DailyDialogスコアリング中、`train_000482#3` と `train_000482#4` でcontent filterが発生したため、安全化入力で再試行した。
- なぜその操作が必要だったか:
  - content filter対象を単純にスキップせず、会話戦略評価を可能な限り継続するため。
- 代替案があったか:
  - スキップまたはnegativeフォールバックがあったが、品質維持のため安全化再評価を優先した。
- 実行したコマンド:
  - `python3 -m tools.score_dialogue_with_transition_bayes_model --input data/dailydialog_for_scoring_500.jsonl --bayes-model artifacts/bayes_models/generated_transition_bayes_model.json --output artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl --model gpt-5.4 --workers 4`
- 変更前後の要約:
  - 変更前: content filter対象ターンは評価不能。
  - 変更後: 安全化入力で再評価を試み、成功時は通常のベイズ更新に使う。
- リスクや注意点:
  - content filter再試行のレコードは、DPO採用前に必要に応じて監査する。

## 2026-06-02 16:21 JST頃

- 対象ファイル:
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500.jsonl`
  - `artifacts/scored_dialogues/dailydialog_transition_scored_500_clean.jsonl`
- 実行した操作:
  - スコアリング完了後、元出力に重複行があることを検出した。
  - 元ファイルは削除せず、`conversation_id` と `turn_index` の組で最後の結果を採用したクリーン版JSONLを別ファイルとして作成した。
- なぜその操作が必要だったか:
  - 入力候補は3165件だが、元出力には重複込みで3873行あり、そのまま抽出すると同じ応答が重複採用されるリスクがあったため。
  - DPOデータの品質と再現性を保つには、1応答候補1スコアに正規化する必要があったため。
- 代替案があったか:
  - 元ファイルを直接上書きする案があったが、監査可能性を残すため採用しなかった。
  - 重複を含んだまま抽出する案があったが、DPOサンプルの偏りにつながるため採用しなかった。
- 実行したコマンド:
  - `python3 - <<'PY' ...重複除去スクリプト... PY`
- 変更前後の要約:
  - 変更前: `dailydialog_transition_scored_500.jsonl` は重複込みで3873行、ユニーク3165件。
  - 変更後: `dailydialog_transition_scored_500_clean.jsonl` はユニーク3165件。
- リスクや注意点:
  - 同じキーの重複では最後に保存された結果を採用した。再開・並列処理後の最新結果を優先する意図。
  - 元ファイルは削除していないため、必要なら後で重複発生箇所を検証できる。

## 2026-06-02 16:30 JST頃

- 対象ファイル:
  - `tools/translate_and_generate_dpo.py`
- 実行した操作:
  - 日本語DPO生成に `--workers` を追加し、サンプル単位で並列生成できるようにした。
  - 採用されたDPOレコードを即時JSONL追記する途中保存処理を追加した。
  - 既存出力がある場合、採用済みの `source_dialogue_id` と `turn_index` をスキップして再開できるようにした。
- なぜその操作が必要だったか:
  - DPO生成は翻訳、chosen再スコアリング、複数rejected生成、rejected再スコアリングを含み、1件あたりのAPI呼び出し回数が多く時間がかかるため。
  - Codexの時間制限が近づいており、ユーザーの通常ターミナルで効率よく再開できるようにする必要があったため。
  - 途中停止時に採用済みDPOサンプルが失われるリスクを下げるため。
- 代替案があったか:
  - 逐次処理を継続する案があったが、時間制約が大きいため採用しなかった。
  - rejected候補数や再スコアリングを減らす案があったが、DPO品質低下につながるため採用しなかった。
- 実行したコマンド:
  - `python3 -B -m pytest -p no:cacheprovider -q`
- 変更前後の要約:
  - 変更前: DPO生成は逐次処理で、完了後にまとめてJSONLを書き出していた。
  - 変更後: `--workers` によりサンプル単位で並列処理でき、採用済みDPOは即時追記される。完了時には最終JSONLとmanifestも書き出す。
- リスクや注意点:
  - 並列化によりAPI呼び出し数の同時実行が増えるため、レート制限やコストに注意する。
  - 品質を落とさないため、翻訳・rejected生成・再スコアリング・score_gap判定の内容は変更していない。
  - `--workers 4` 程度から始めるのが安全。レート制限が出る場合は `--workers 2` に下げる。

## 2026-06-03 06:54:07 JST: DPO生成時のcontent_filter停止対策

- 実行日時:
  - 2026-06-03 06:54:07 JST
- 対象ファイル:
  - `tools/translate_and_generate_dpo.py`
  - `tests/test_dailydialog_pipeline.py`
  - `audit_log.md`
- 実行した操作:
  - 翻訳・rejected生成API呼び出しでAzure OpenAIのcontent_filterが発生した場合、そのサンプルだけを `content_filter_generation` としてスキップし、DPO生成全体は継続するようにした。
  - 日本語chosen/rejectedの再スコアリングでは、既存の状態遷移ベイズスコアリング側のcontent_filter安全化再試行・フォールバック処理を使うようにした。
  - content_filterで1件が止まっても、次の候補からDPO生成を続けられることをテストで確認した。
- なぜその操作が必要だったか:
  - `run_dpo_pipeline.sh` 実行中、DailyDialog由来の特定サンプルがAzure OpenAIのcontent_filterに誤検出され、並列workerの例外が全体プロセスを停止させたため。
  - 研究用途では大量データ全体を評価することが重要だが、API側が拒否したサンプルを無理に採用するとDPOデータ品質と再現性が落ちるため、該当サンプルだけを明示的に除外する必要があった。
- 代替案があったか:
  - content_filterを誘発した元英文を安全化して翻訳・rejected生成を再試行する案があった。ただし、翻訳対象そのものを書き換えるとchosenの意味や会話戦略が変わる可能性があるため、DPO品質を優先して採用しなかった。
  - worker数を1に下げる案があったが、content_filter自体は逐次処理でも発生するため根本解決にならない。
- 実行したコマンド:
  - `python3 -m py_compile tools/translate_and_generate_dpo.py tests/test_dailydialog_pipeline.py`
  - `python3 -B -m pytest -p no:cacheprovider -q`
- 変更前後の要約:
  - 変更前: 翻訳・rejected生成APIでcontent_filterが発生すると、`future.result()` が例外を投げ、DPO生成全体が停止していた。
  - 変更後: content_filterが発生したサンプルは `skip content_filter generation` として記録され、他のサンプルのDPO生成は継続する。再スコアリング側のcontent_filterは既存の安全化再試行・フォールバックを使う。
- リスクや注意点:
  - content_filter対象サンプルはDPOデータから除外されるため、最終件数が少し減る可能性がある。
  - content_filterの発生が多い場合は、選択済み英語候補数を増やすか、抽出閾値を調整して十分なDPO件数を確保する必要がある。
  - content_filter除外はデータセットのスキップに該当するため、ログで件数を確認する。

## 2026-06-03 09:30:00 JST: 実行時audit_log自動追記の追加

- 対象ファイル:
  - `tools/audit_logging.py`
  - `tools/score_dialogue_with_transition_bayes_model.py`
  - `tools/translate_and_generate_dpo.py`
  - `audit_log.md`
- 実行した操作:
  - スコアリング完了時とDPO生成完了時に、重要操作の要約を `audit_log.md` へ自動追記する処理を追加した。
- なぜその操作が必要だったか:
  - content_filter、スキップ、閾値による除外など、研究上重要な判断を後から追跡できるようにするため。
- 代替案があったか:
  - ターミナルログだけに残す案があったが、後から見返す研究記録としては埋もれやすいため採用しなかった。
  - 個別レコード全文をaudit_log.mdへ保存する案があったが、ログ肥大化とデータ本文露出を避けるため採用しなかった。
- 実行したコマンド:
  - `python3 -B -m pytest -p no:cacheprovider -q`
- 変更前後の要約:
  - 変更前: content_filterやskip理由はターミナルログ、JSONLの `llm_retry` / `llm_error`、manifestには残るが、`audit_log.md` へは自動追記されなかった。
  - 変更後: 大規模スコアリングでは入力件数、出力件数、content_filter安全化再試行件数、フォールバック件数を `audit_log.md` に記録する。
  - 変更後: DPO生成では採用件数、各skip理由、使用モデル、閾値、ベイズモデルバージョンを `audit_log.md` に記録する。
- リスクや注意点:
  - 個別レコード本文は `audit_log.md` に保存しないため、詳細確認には対応するJSONL、manifest、パイプラインログを併用する必要がある。
  - 既存の `tools.audit_dpo_preferences` は現在の一括実行スクリプトでは使わないため、品質監査APIによるサンプル除外は行われない。

## 2026-06-03 09:22:52 JST: DPOパイプラインwatchdogイベント

- 対象ファイル:
  - `./run_dpo_pipeline_reminiscence_2000.sh`
  - `artifacts/scored_dialogues/dailydialog_transition_scored_reminiscence_5000_to_2000.jsonl`
  - `logs/dpo_pipeline_reminiscence_5000_to_2000_watchdog_20260603_092252.log`
- 実行した操作:
  - watchdog付きでDPOパイプラインを開始した。
- なぜその操作が必要だったか:
  - 席を離している間にAPI待ちやcontent_filter再試行で処理が停止したままになるリスクを下げるため。
- 代替案があったか:
  - 手動監視する案があったが、長時間不在時に停止を検出できないため採用しなかった。
- 実行したコマンド:
  - `./run_dpo_pipeline_reminiscence_2000_watchdog.sh`
- 変更前後の要約:
  - 進捗停止判定秒数: 600
  - 最大再起動回数: 20
  - 初回SCORING_WORKERS: 4
  - 再起動時SCORING_WORKERS: 2
- リスクや注意点:
  - 再起動時、実行中のパイプラインプロセスグループへTERM/KILLを送る。保存済みJSONLは再実行時にスキップされる。
  - 学習ステップ中はスコア済みJSONLの行数が増えないため、watchdogはスコアリング未完了時だけ停止判定する。

## 2026-06-03 09:44:02 JST: DPOパイプラインwatchdogイベント

- 対象ファイル:
  - `./run_dpo_pipeline_reminiscence_2000.sh`
  - `artifacts/scored_dialogues/dailydialog_transition_scored_reminiscence_5000_to_2000.jsonl`
  - `logs/dpo_pipeline_reminiscence_5000_to_2000_watchdog_20260603_092252.log`
- 実行した操作:
  - スコアリング行数が600秒間増えなかったため、パイプラインを再起動した。再起動回数: 1
- なぜその操作が必要だったか:
  - 席を離している間にAPI待ちやcontent_filter再試行で処理が停止したままになるリスクを下げるため。
- 代替案があったか:
  - 手動監視する案があったが、長時間不在時に停止を検出できないため採用しなかった。
- 実行したコマンド:
  - `./run_dpo_pipeline_reminiscence_2000_watchdog.sh`
- 変更前後の要約:
  - 進捗停止判定秒数: 600
  - 最大再起動回数: 20
  - 初回SCORING_WORKERS: 4
  - 再起動時SCORING_WORKERS: 2
- リスクや注意点:
  - 再起動時、実行中のパイプラインプロセスグループへTERM/KILLを送る。保存済みJSONLは再実行時にスキップされる。
  - 学習ステップ中はスコア済みJSONLの行数が増えないため、watchdogはスコアリング未完了時だけ停止判定する。

## 2026-06-03 09:45:00 JST: watchdog再起動時のプロセスグループkill強化

- 対象ファイル:
  - `run_dpo_pipeline_reminiscence_2000_watchdog.sh`
  - `audit_log.md`
- 実行した操作:
  - watchdogがパイプラインを再起動する際、実行中の親プロセスだけでなく子プロセスも含めて停止できるように、`setsid` で独立プロセスグループとして起動する処理を追加した。
  - 停止時は `kill -TERM -- -PID` を送り、残っていれば `kill -KILL -- -PID` を送るようにした。
- なぜその操作が必要だったか:
  - 再起動時に既存のPython/API処理が残ると、プロセスが増え続けてAPI負荷・料金・出力JSONL競合のリスクがあるため。
- 代替案があったか:
  - 親PIDだけをkillする案があったが、子プロセスが残る可能性があるため採用しなかった。
  - 手動で `pgrep` と `kill` を確認する案があったが、不在中の自動復旧には不十分なため採用しなかった。
- 実行したコマンド:
  - `bash -n run_dpo_pipeline_reminiscence_2000_watchdog.sh`
- 変更前後の要約:
  - 変更前: バックグラウンド起動したパイプラインのPIDへkillを試みていたが、子プロセスまで確実に止められる保証が弱かった。
  - 変更後: `setsid` が使える環境ではパイプラインを独立プロセスグループ化し、再起動時にグループ全体を停止する。
- リスクや注意点:
  - 再起動判定時には実行中パイプライン全体を止めるため、最後に処理中だった数件は再実行される可能性がある。
  - JSONLは完了済みレコードのみ追記され、再実行時に処理済みキーをスキップする設計なので、基本的には途中再開できる。

## 2026-06-03 09:55:52 JST: DPOパイプラインwatchdogイベント

- 対象ファイル:
  - `./run_dpo_pipeline_reminiscence_2000.sh`
  - `artifacts/scored_dialogues/dailydialog_transition_scored_reminiscence_5000_to_2000.jsonl`
  - `logs/dpo_pipeline_reminiscence_5000_to_2000_watchdog_20260603_095552.log`
- 実行した操作:
  - watchdog付きでDPOパイプラインを開始した。
- なぜその操作が必要だったか:
  - 席を離している間にAPI待ちやcontent_filter再試行で処理が停止したままになるリスクを下げるため。
- 代替案があったか:
  - 手動監視する案があったが、長時間不在時に停止を検出できないため採用しなかった。
- 実行したコマンド:
  - `./run_dpo_pipeline_reminiscence_2000_watchdog.sh`
- 変更前後の要約:
  - 進捗停止判定秒数: 600
  - 最大再起動回数: 20
  - 初回SCORING_WORKERS: 4
  - 再起動時SCORING_WORKERS: 4
- リスクや注意点:
  - 再起動時、実行中のパイプラインプロセスグループへTERM/KILLを送る。保存済みJSONLは再実行時にスキップされる。
  - 学習ステップ中はスコア済みJSONLの行数が増えないため、watchdogはスコアリング未完了時だけ停止判定する。

## 2026-06-03 10:00:00 JST: DPOパイプラインログ出力先の分離

- 対象ファイル:
  - `run_dpo_pipeline_reminiscence_2000.sh`
  - `run_dpo_pipeline_reminiscence_2000_watchdog.sh`
  - `run_dpo_pipeline.sh`
  - `audit_log.md`
- 実行した操作:
  - 通常DPOパイプラインログの保存先を `logs/dpo_pipeline/` に変更した。
  - watchdogログの保存先を `logs/dpo_pipeline_watchdog/` に変更した。
  - watchdogから通常パイプラインを起動する際、`PIPELINE_LOG_DIR` を明示的に引き継ぐようにした。
- なぜその操作が必要だったか:
  - 通常pipelineログとwatchdogログが同じ `logs/` 直下に混在し、後から確認しにくくなっていたため。
- 代替案があったか:
  - 既存ログを移動するだけの案があったが、今後の実行で再び混在するためコード側の出力先を変更した。
- 実行したコマンド:
  - `bash -n run_dpo_pipeline_reminiscence_2000.sh && bash -n run_dpo_pipeline_reminiscence_2000_watchdog.sh && bash -n run_dpo_pipeline.sh`
- 変更前後の要約:
  - 変更前: `logs/dpo_pipeline_*.log` と `logs/dpo_pipeline_*_watchdog_*.log` が `logs/` 直下に保存されていた。
  - 変更後: 通常ログは `logs/dpo_pipeline/`、watchdogログは `logs/dpo_pipeline_watchdog/` に保存される。
- リスクや注意点:
  - 既に移動済みの過去ログは変更していない。
  - 実行時に `PIPELINE_LOG_DIR` や `WATCHDOG_LOG_DIR` を指定すれば、保存先を上書きできる。

## 2026-06-03 10:20:00 JST: Oracle正解応答100点満点評価パイプラインの追加

- 対象ファイル:
  - `tools/run_oracle_evaluation.py`
  - `configs/evaluation_prompts/reminiscence_oracle_eval_v1.jsonl`
  - `run_oracle_evaluation_reminiscence.sh`
  - `tests/test_oracle_evaluation.py`
  - `audit_log.md`
- 実行した操作:
  - GPT-5.4-proをOracleとして使い、評価promptごとに理想応答を生成し、その理想応答を100点満点の正解としてbase/DPO応答を採点するCLIを追加した。
  - Oracle理想応答生成・採点時には、生成済み状態遷移ベイズモデルに加えて、小コーパス本文抜粋も参照するようにした。
  - 固定評価promptセットと、推奨設定で実行するシェルスクリプトを追加した。
- なぜその操作が必要だったか:
  - 教授から提案された「正解を満点とするOracle評価」を、学習済みモデルとベースモデルの比較評価として再現可能に実行するため。
  - 小コーパスを正解基準とする研究目的に合わせ、Oracleが小コーパス本文とそこから生成されたベイズモデルの両方を参照できるようにするため。
- 代替案があったか:
  - GPT-5.4-proに単純なペア比較だけをさせる案があったが、「正解を満点とするOracle」という説明には弱いため採用しなかった。
  - 追加でOracle専用モデルを学習する案があったが、現段階ではコストと時間が大きく、GPT-5.4-proを仮の高品質Oracleとして使う方が現実的なため採用しなかった。
- 実行したコマンド:
  - `python3 -m py_compile tools/run_oracle_evaluation.py`
  - `bash -n run_oracle_evaluation_reminiscence.sh`
  - `python3 -B -m pytest -p no:cacheprovider -q tests/test_oracle_evaluation.py`
  - `python3 -m tools.run_oracle_evaluation --dry-run --max-prompts 3`
  - `python3 -B -m pytest -p no:cacheprovider -q`
- 変更前後の要約:
  - 変更前: 学習後モデルの比較は主に人がチャットUIで確認する形で、Oracle正解応答に対する100点満点評価はなかった。
  - 変更後: base/DPO応答、Oracle理想応答、100点満点採点、勝敗、カテゴリ別summaryをJSONL/JSONで保存できる。
  - 変更後: Oracle評価promptでは「回想法」と明示せず、ユーザー発話に昔の経験を自然に含めることで、ベースモデルにも公平な比較条件を保つ。
- リスクや注意点:
  - OracleはGPT-5.4-proであり、人手評価の完全な代替ではない。最終的には一部サンプルを人手で確認すると説得力が上がる。
  - Oracleが小コーパス本文を参照するため、評価prompt数が増えるとAPIコストが増える。まずは20件の初期セットまたは50件程度で確認する。

## 2026-06-03 10:35:00 JST: 実行用shファイルのscriptsディレクトリ集約

- 対象ファイル:
  - `scripts/run_dpo_pipeline.sh`
  - `scripts/run_dpo_pipeline_reminiscence_2000.sh`
  - `scripts/run_dpo_pipeline_reminiscence_2000_watchdog.sh`
  - `scripts/run_oracle_evaluation_reminiscence.sh`
  - `audit_log.md`
- 実行した操作:
  - ルート直下にあった実行用 `.sh` ファイルを `scripts/` ディレクトリへ移動した。
  - 各スクリプトの冒頭でリポジトリルートへ `cd` する処理を追加し、`scripts/` 配下から実行しても既存の相対パスが壊れないようにした。
  - watchdogの既定 `PIPELINE_SCRIPT` を移動後の `scripts/run_dpo_pipeline_reminiscence_2000.sh` に合わせた。
- なぜその操作が必要だったか:
  - ルート直下に複数の実行用shファイルが並び、ログや実験用スクリプトの管理が見にくくなっていたため。
- 代替案があったか:
  - ルート直下に残してREADMEだけで説明する案があったが、ファイル一覧の見通しが改善しないため採用しなかった。
  - ルートに互換用symlinkを残す案があったが、整理目的に反するため今回は採用しなかった。
- 実行したコマンド:
  - `mkdir -p scripts && mv run_dpo_pipeline.sh run_dpo_pipeline_reminiscence_2000.sh run_dpo_pipeline_reminiscence_2000_watchdog.sh run_oracle_evaluation_reminiscence.sh scripts/`
  - `bash -n scripts/run_dpo_pipeline.sh && bash -n scripts/run_dpo_pipeline_reminiscence_2000.sh && bash -n scripts/run_dpo_pipeline_reminiscence_2000_watchdog.sh && bash -n scripts/run_oracle_evaluation_reminiscence.sh`
- 変更前後の要約:
  - 変更前: `run_*.sh` がリポジトリルート直下に保存されていた。
  - 変更後: 実行用shは `scripts/` に集約され、ルートから `./scripts/<script>.sh` として実行する。
- リスクや注意点:
  - 旧コマンド `./run_dpo_pipeline_reminiscence_2000_watchdog.sh` は使えない。今後は `./scripts/run_dpo_pipeline_reminiscence_2000_watchdog.sh` を使う。
  - スクリプト内部でリポジトリルートへ移動するため、どのディレクトリから起動しても基本的には同じ出力先になる。
