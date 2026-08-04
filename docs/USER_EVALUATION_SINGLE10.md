# 3データセット単一10問ユーザ評価

## 目的と解釈

ESConv、MathDial、MediTODの人手評価を、A/B各10問ではなく、全参加者が
同じ10問へ回答する形式で実施する。選定項目はOracle評価でBASiSが高く、
比較モデルとの差があり、かつ人が応答差を読み取りやすい候補へ富化している。
このため、test全体の無条件な主評価ではなく、BASiSの得意場面を対象にした
副次的人手評価として報告する。

モデル応答は生成時の内容から編集しない。途中切れ、文字化け、内容矛盾、
3応答の差が小さい項目は除外し、別の評価済み候補へ差し替える。

## 画面構成

- 最初の画面に、評価対象の会話スタイル、良い例、良くない例、画面の見方を示す。
- 評価画面の左側に会話履歴、応答A〜C、評価例を表示する。
- 長い会話は直近6発話を表示し、前半は必要なときだけ展開できる。
- 右側だけをスクロールし、各応答の7段階評価、最終選択、選択理由を入力する。
- 次の問題へ移ると右側のスクロール位置を先頭へ戻す。

## 成果物

### ESConv

- 公開item: `artifacts/user_eval/google_forms/esconv_human_reviewed_likert_single10_v8/experiment_a/form_items_public.jsonl`
- 非公開正解表: `artifacts/user_eval/google_forms/esconv_human_reviewed_likert_single10_v8/experiment_a/answer_key_private.csv`
- 非公開選定監査: `artifacts/user_eval/google_forms/esconv_human_reviewed_likert_single10_v8/selection_audit_private.md`
- 回答DB: `artifacts/user_eval/web/esconv_likert_single10_responses.sqlite3`

### MathDial

- 公開item: `artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/user_eval_v3_single10/experiment_a/form_items_public.jsonl`
- 非公開正解表: `artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/user_eval_v3_single10/private_answer_key.jsonl`
- 非公開選定監査: `artifacts/mathdial_wildchat/evaluation_rechecks/mathdial_v6_instruction_outcome_selected_top100_v1/user_eval_v3_single10/selection_review_private.md`
- 回答DB: `artifacts/user_eval/web/mathdial_likert_v3_single10_responses.sqlite3`

### MediTOD

- 公開item: `artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v3_single10/experiment_a/form_items_public.jsonl`
- 非公開正解表: `artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v3_single10/private_answer_key.jsonl`
- 非公開選定監査: `artifacts/meditod_wildchat/runs/meditod_wildchat_gpt56_v2/user_eval_v3_single10/selection_review_private.md`
- 回答DB: `artifacts/user_eval/web/meditod_likert_v3_single10_responses.sqlite3`

非公開正解表と選定監査は参加者へ共有しない。旧A/B版の成果物と回答DBは
履歴として残し、単一10問版へ混ぜない。

## 準備

```bash
./scripts/prepare_esconv_likert_single10.sh
./scripts/prepare_mathdial_likert_single10.sh
./scripts/prepare_meditod_likert_single10.sh
```

## 起動

別々のtmux paneで起動する。

```bash
PUBLIC_HOST=192.168.1.17 PORT=8503 \
  ./scripts/run_esconv_likert_single10_web.sh

PUBLIC_HOST=192.168.1.17 PORT=8504 \
  ./scripts/run_mathdial_likert_single10_web.sh

PUBLIC_HOST=192.168.1.17 PORT=8505 \
  ./scripts/run_meditod_likert_single10_web.sh
```

各アプリはA/B指定のない共通URLを表示する。回答は各データセット専用SQLiteへ
逐次保存され、同じ氏名で再度開くと未回答の問題から再開する。

## 最新結果の出力

回答受付中でも、次のコマンドで3データセットのSQLiteスナップショットを
CSVへ再集計できる。

```bash
./scripts/update_single10_user_eval_results.sh
```

各データセットの`artifacts/user_eval/results/<dataset>_single10/`へ、
`responses_long_private.csv`、`axis_model_summary.csv`、`friedman.csv`、
`holm_posthoc.csv`、`final_choice_counts.csv`、`metadata.json`を書き出す。
同じコマンドを再実行すると、その時点の最新回答でCSVを更新する。
