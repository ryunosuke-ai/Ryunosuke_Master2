# BASiS vs Random ユーザ評価設計

## 実験目的

Oracle評価を補足する追加の人手評価として、BASiSで選別したデータを用いて学習したモデルの応答が、Randomで選別したデータを用いて学習したモデルの応答よりも、目的コーパス由来の支援対話スタイルになっているかを確認する。

この評価は中間発表における主実験ではなく、LLM-as-a-judgeのOracle評価結果を人間評価で補強する短期間の研究室内評価として位置づける。

## 比較対象

- BASiS/Bayes-DPO: `responses.jsonl` の `base_response`
- Random-DPO: `responses.jsonl` の `dpo_response`

実装上の互換性により、既存Oracle評価では `base` field がBASiS/Bayes-DPO、`dpo` field がRandom-DPOを表す。評価者にはこの対応を表示しない。

主な入力ファイル:

- Oracle比較run: `artifacts/evaluations/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/`
- 応答本文: `artifacts/evaluations/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/responses.jsonl`
- Oracle判定: `artifacts/evaluations/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/judgments.jsonl`
- 実験設定: `artifacts/evaluations/oracle_eval_runs/esconv_5000_to_2000_bayes_vs_random2500_oracle_esconv_v3_strategy/manifest.json`
- 元prompt: `configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl`

## 評価形式

各itemで、評価用プロンプトと2つの匿名応答を表示する。

- 表示名は `Model A` / `Model B`
- BASiS/Randomの対応は評価者に見せない
- A/B表示順はseed固定でランダム化する
- 評価は5つの観点ごとの5段階A/B比較
- 任意コメント欄を設ける

各観点の評価尺度:

| 値 | 意味 |
|---:|---|
| 1 | Aの方がかなり当てはまる |
| 2 | Aの方がやや当てはまる |
| 3 | どちらも同程度 |
| 4 | Bの方がやや当てはまる |
| 5 | Bの方がかなり当てはまる |

## 評価観点

評価者には、次の5つが今回の評価観点であると説明し、各観点についてModel A / Model Bのどちらがより当てはまるかを選んでもらう。GUI上の会話スタイル説明では、評価者が余計な先入観を持たないように元コーパス名は出さず、「カウンセリング場面で見られる相談支援らしい会話スタイル」として説明する。

- 気持ちの受け止め: 相談者の気持ちをより受け止めているのはどちらか
- 助言のタイミング: 助言や提案に進むタイミングがより自然なのはどちらか
- 話への合い方: 相手の話に合った聞き返しや整理ができているのはどちらか
- 温かさ: 温かく、相談者が話し続けやすいのはどちらか
- 会話の前進: 必要に応じて会話を前に進めているのはどちらか

## データ選定方法

評価件数は30件とする。今回はBASiSの有効性が人手評価でも確認できるかを見る補足評価なので、完全に無作為な代表サンプルではなく、Oracle評価でBASiS優位と予想されるitemを多めに含める。

既定設定:

- seed: `20260619`
- total: 30件
- `oracle_basis_win`: 20件
- `oracle_random_win`: 5件
- `oracle_close`: 5件
- close判定: `abs(BASiS score - Random score) < 3.0`
- カテゴリ分布: 可能な限り10カテゴリ各3件

stratum定義:

- `oracle_basis_win`: `BASiS score - Random score >= 3.0`
- `oracle_random_win`: `BASiS score - Random score <= -3.0`
- `oracle_close`: `abs(BASiS score - Random score) < 3.0`

`scripts/prepare_user_eval_items.py` は、seedと選定ロジックからitemを再生成し、次を保存する。

- 選定方法
- seed
- item_id
- category
- stratum
- Oracle上の勝敗
- BASiS score
- Random score
- score gap
- 元データパス
- A/B表示順

出力先:

- `artifacts/user_eval/items/user_eval_items.jsonl`
- `artifacts/user_eval/items/selection_manifest.json`
- `artifacts/user_eval/items/selected_items.csv`

## GUI仕様

Streamlitアプリ `apps/user_eval_app.py` を使う。

開始画面:

- 氏名または参加者IDの入力
- 同意確認
- 研究の簡単な説明
- 評価目的
- 操作手順
- 評価観点
- 所要時間の目安
- モデル名が匿名化されていること
- 回答は研究目的で集計すること
- 個人情報の扱い

評価画面:

- 現在の評価番号と進捗バー
- 目的の会話スタイル説明カード
- 評価中にも確認できる評価観点カード
- 会話履歴
- 評価用プロンプト
- Model A / Model Bの応答カード
- 5つの評価観点ごとの5段階ラジオボタン
- 任意コメント
- 前へ戻るボタン
- 未入力時は次へ進めない
- 完了後に「回答ありがとうございました」を表示

同時アクセス対応:

- `session_id` をUUIDベースで一意生成する
- 回答ファイルは `{participant_id}_{session_id}.jsonl`
- 参加者IDが同じでもsession_idが異なれば別ファイルになる
- 未回答itemは新規追加し、回答済みitemを修正した場合は同じ行を置換する
- 同じsession内で同じitemを二重保存しない
- 戻って修正した場合は、同じitemの既存回答を置換する
- 保存時にファイルロックを使い、一時ファイル経由でJSONLを更新する

氏名と匿名ID:

- `participant_name`: 入力された氏名または参加者ID。ローカルの生回答JSONLにのみ保存する
- `participant_id`: `participant_name` から作るハッシュID。集計・発表用に使う
- `session_id`: セッションごとの一意ID

発表用CSV、Markdown、グラフには `participant_name` を出力しない。

## 保存形式

回答JSONLの主な項目:

- `participant_name`
- `participant_id`
- `session_id`
- `item_id`
- `category`
- `stratum`
- `history`
- `prompt`
- `model_a_response`
- `model_b_response`
- `model_a_source`
- `model_b_source`
- `displayed_order`
- `basis_position`
- `random_position`
- `axis_ratings`
- `comment`
- `timestamp`
- `revision`
- `created_at`
- `updated_at`

`axis_ratings` は5つの評価観点ごとの1〜5の値を保存する。`model_a_source` / `model_b_source` は保存するが、GUIには表示しない。

## 分析方法

`scripts/analyze_user_eval_results.py` で複数JSONLを統合し、A/B表示順を補正して、各評価軸をBASiS基準スコアに変換する。

各軸のBASiS基準スコア:

- BASiSがかなり良い: `+2`
- BASiSがやや良い: `+1`
- 同程度: `0`
- Randomがやや良い: `-1`
- Randomがかなり良い: `-2`

総合スコアは、Oracle評価のweighted overallに対応する考え方で5軸を重み付き合成する。

| 評価観点 | 重み | 対応するOracle評価の考え方 |
|---|---:|---|
| 気持ちの受け止め | 0.25 | 感情を受け止めているか |
| 助言のタイミング | 0.20 | 早すぎる助言を避けているか |
| 話への合い方 | 0.35 | 文脈に合った聞き返し・整理ができているか |
| 温かさ | 0.15 | 温かく支援的で、全体として助けになるか |
| 会話の前進 | 0.05 | 必要に応じて会話を前に進めているか |

出力:

- `artifacts/user_eval/results/normalized_responses.csv`
- `artifacts/user_eval/results/summary.csv`
- `artifacts/user_eval/results/participant_summary.csv`
- `artifacts/user_eval/results/item_summary.csv`
- `artifacts/user_eval/results/report.md`
- `artifacts/user_eval/results/figures/*.png`
- `artifacts/user_eval/results/figures/*.svg`

集計内容:

- BASiS勝ち / Random勝ち / Tie件数
- BASiS勝率 / Random勝率 / Tie率
- 5軸すべての5段階評価の分布
- 重み付きBASiS基準スコアの平均・標準偏差・標準誤差
- 軸別BASiS基準スコアの平均・勝率
- 95%信頼区間
- 参加者ごとの回答数
- itemごとの結果
- ties除外のexact sign test
- Markdownレポート
- 発表用グラフ

発表用グラフ:

- `win_rate_bar`
- `rating_distribution`
- `basis_score_distribution`
- `axis_mean_scores`
