# ESConv 7段階ユーザ評価

## 評価形式

同一の日本語会話履歴に対するBase、BASiS、Random-DPOの応答を匿名化し、
各応答を7軸で独立に評価する。その後、3応答のうち相談支援スタイルとして
最もふさわしい応答を1つ選ぶ。

各応答の尺度:

| 値 | 意味 |
|---:|---|
| 1 | 全く当てはまらない |
| 2 | 当てはまらない |
| 3 | あまり当てはまらない |
| 4 | どちらともいえない |
| 5 | やや当てはまる |
| 6 | 当てはまる |
| 7 | 非常によく当てはまる |

評価文:

1. 相談者を支える応答として、全体的に良い。
2. 相談者の気持ちを受け止め、やさしく話している。
3. これまでの会話に合った、支える立場の話し方を続けている。
4. 相手の話を理解・整理しようとし、指示や結論をすぐに押しつけていない。
5. この会話の段階に対して、助言や提案を出すタイミングが適切である。
6. これまでの話の内容によく合っている。
7. 日本語の会話として自然で読みやすい。

最後の質問:

> 3つの応答のうち、相談者の気持ちを受け止め、会話の状況に合わせて
> 支える応答として、最もふさわしいものはどれですか。

選択肢は`応答A / 応答B / 応答C / ほぼ同じ / 判断できない`とする。

## 選定した20件

100件のOracle評価結果から、代表5軸におけるBASiS平均とBase・Random-DPOの
良い方の平均との差を算出する。ユーザ評価でモデル差を確認するため、この差が
大きい上位20件を選ぶ。選定20件はBASiS代表5軸平均8.5以上、最良controlとの
差が0.6点以上である。

これはOracle結果を見た後の対象化選定である。この20件で計算したOracleの
有意差は選定条件付きの事後診断であり、人手評価の有意差ではない。
論文では「OracleでBASiS優位が確認された場面に限定したユーザ評価」と明記し、
ESConv全体の無条件な結果へ一般化しない。

## 旧20件一括版

以下はカテゴリ均等で選んだ旧20件を、1人へすべて提示する監査用の旧版である。
現在のユーザ評価には使用しない。

```bash
python3 scripts/prepare_esconv_google_form_likert_eval.py
```

出力:

```text
artifacts/user_eval/google_forms/esconv_oracle_enriched_likert_v2/
  questionnaire_spec.json
  selection_manifest.json
  selection_conditioned_diagnostics.json
  form_version_a/
  form_version_b/
  form_version_c/
```

各フォーム版の`create_google_form.gs`をGoogle Apps Scriptへ貼り付け、
`createEsconvLikertForm`を実行する。参加者はA/B/Cのいずれか1版だけへ
割り付ける。3版を通して各モデルの表示位置が均衡する。

## 推奨する2フォーム構成

参加者の負担を下げるため、実施時は20件を10件ずつの評価実験A/Bへ分ける。
識別力を優先した20件のカテゴリ構成がA/Bでできるだけ近くなり、Oracle優位度
の平均が一致するよう分割する。2実験の和が選定した20件になる。

```bash
python3 scripts/prepare_esconv_google_form_likert_blocks.py
```

出力:

```text
artifacts/user_eval/google_forms/esconv_discriminative_likert_two_forms_v5/
  block_manifest.json
  participant_assignment_template.csv
  experiment_a/
    create_google_form.gs
    form_items_public.jsonl
    private_model_mapping.jsonl
  experiment_b/
    create_google_form.gs
    form_items_public.jsonl
    private_model_mapping.jsonl
```

参加者を実験A/Bへできるだけ同数に割り当て、各参加者は一方だけを評価する。
氏名と回答を保存することを同意文に明記し、回答スプレッドシートは研究担当者
だけがアクセスできるようにする。
各フォーム内では、各モデルが応答A/B/Cの各位置へ3回または4回現れるよう
固定し、表示位置の偏りを抑える。1人あたりは10件、210個のLikert評定と
10個の総合選択になる。実験A/Bの結果は、共通する軸とモデル条件を使って
統合解析できるが、参加者が異なるため参加者とitem IDを含む混合効果
モデルを使う。

## 解析

- 代表5軸を主評価とし、内容保持・自然さを品質制約として分けて報告する。
- 各軸と代表5軸平均について、3モデルの対応あり比較を行う。
- 参加者とitemの反復を扱える混合効果モデルを主解析とする。
- 補助解析としてFriedman検定、有意時の対応あり置換検定またはWilcoxon検定、
  Holm補正、効果量、bootstrap 95%信頼区間を出す。
- 最後の総合選択はモデル別選択率を出し、参加者・itemを考慮した比較を行う。
- `ほぼ同じ`は同率、`判断できない`は当該比較の欠測として回答収集前に固定する。
- 推奨する2実験構成でも1人220判断になるため、少人数で所要時間と疲労を
  確認してから本実施する。
