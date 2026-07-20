# ESConv Google Formユーザ評価

## 位置づけ

ESConvのユーザ評価では、同一の日本語会話履歴に対するBase、BASiS、
Random-DPOの応答を匿名化して比較する。

Oracle評価でBASiSが高かったitemだけを選ぶと選択バイアスが生じるため、
次の2セットを分離する。

- `main_confirmatory`: Oracleスコアを使わず、100件からカテゴリ層化で20件を選ぶ。
  論文の主たる人手評価にはこちらを使用する。
- `oracle_enriched_exploratory`: 代表5軸でBASiSが高かった20件。
  探索的な事例分析・エラー分析に限定し、無条件の主検定には使用しない。

1件単独について「有意差がある」とは判定できない。有意差は、複数item・
複数参加者から得た対応あり回答を集計した後に検定する。

## 質問

1件につき次の7問を使用する。最初の5問がESConv代表軸、後ろの2問が
スタイル転移の品質制約である。

1. 相談している人を支える応答として、全体的に最も良いのはどれですか。
2. 相談している人の気持ちを受け止め、やさしく話しているのはどれですか。
3. これまでの会話に合った、支える立場の話し方を続けているのはどれですか。
4. 相手の話を理解・整理しようとし、すぐに指示や結論を押しつけていないのは
   どれですか。
5. この会話の段階を考えたとき、助言や提案を出すタイミングが最も適切なのは
   どれですか。
6. これまでの話の内容に最もよく合っているのはどれですか。
7. 日本語の会話として最も自然で読みやすいのはどれですか。

選択肢は共通して、`応答A / 応答B / 応答C / ほぼ同じ / 判断できない`とする。

## フォーム版

各セットにフォームA/B/Cを作る。同一itemでも3版でモデルの表示位置を循環し、
参加者を3版へできるだけ同数に割り付ける。これによりA/B/Cの位置バイアスを
抑える。

参加者へ共有してよいファイル:

- `form_items_public.jsonl`
- `google_form_items.csv`
- `google_form_sections.md`
- `create_google_form.gs`

`private_model_mapping.jsonl`にはモデル対応とOracleスコアが含まれるため、
評価終了まで参加者へ見せない。

## 作成

```bash
python3 scripts/prepare_esconv_google_form_eval.py
```

出力先:

```text
artifacts/user_eval/google_forms/esconv_representative_v1/
  questionnaire_spec.json
  selection_manifest.json
  main_confirmatory/
    form_version_a/
    form_version_b/
    form_version_c/
  oracle_enriched_exploratory/
    form_version_a/
    form_version_b/
    form_version_c/
```

Google Formを自動作成する場合は、対象版の`create_google_form.gs`を
Google Apps Scriptへ貼り付け、`createEsconvEvaluationForm`を実行する。
作成後にログへ編集URLと回答URLが表示される。
「同意しない」を選んだ参加者は評価項目へ進まず、その時点でフォームを
送信する分岐になっている。

## 実施上の注意

- 氏名ではなく研究用参加者IDを使用する。
- フォームA/B/Cの参加者数を均等にする。
- `main_confirmatory`と`oracle_enriched_exploratory`を同じ主検定へ混ぜない。
- 参加者へモデル名、Oracleスコア、選定層を見せない。
- 参加者ごとの回答時間、除外基準、目標参加者数を回答収集前に固定する。
- 7問×20件で140回答となるため、事前に少人数で所要時間と疲労を確認する。

## 集計方針

- `応答A/B/C`は非公開の`private_model_mapping.jsonl`を使い、回答収集後に
  `Base / BASiS / Random-DPO`へ戻す。
- `ほぼ同じ`は同率、`判断できない`はその軸の欠測として、事前に扱いを
  固定する。
- 参加者と評価itemの両方で回答が繰り返されるため、回答を独立な単純票として
  扱わない。主解析では参加者・itemの反復を考慮した混合効果モデル、または
  参加者単位の対応を保った置換検定を使う。
- 代表5軸は主解析、`話の内容への合い方`と`日本語の自然さ`は品質制約として
  分けて報告する。
- 主検定に使うモデル比較、除外規則、多重比較補正を回答収集前に固定する。
