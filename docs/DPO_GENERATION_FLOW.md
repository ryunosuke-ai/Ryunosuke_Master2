# DPO生成処理の流れ

この文書は、ESConv/回想法パイプラインで出る次のようなログの意味を後から確認するためのメモです。

```text
[STEP 5/6] dpo generation: 105/400 (26.2%) accepted=9 skipped=95 train_000217#5
[STEP 5/6] skip high rejected posterior
```

## 1. 入力は「高スコア候補」

DPO生成の入力は、すでにスコアリングと抽出が済んだ候補です。

DailyDialog由来のESConv DPOでは、流れは次の通りです。

```text
DailyDialogの英語対話
  -> ESConvベイズモデルでスコアリング
  -> posteriorが高い応答だけを抽出
  -> 抽出済み応答から日本語DPOを作る
```

つまりDPO生成に入る時点で、元の英語応答は「chosen候補として使えそうな高posterior応答」です。

## 2. DPO生成で作るもの

1件の入力候補から、最終的には次のJSONL 1行を作ろうとします。

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "metadata": {}
}
```

ただし、最初からこの1行を無条件に保存するわけではありません。まずLLMで日本語化とrejected候補生成を行い、その後に再スコアリングして、品質条件を満たしたものだけ採用します。

## 3. chosenとrejectedはどう作るか

`tools/translate_and_generate_dpo.py` のDPO生成では、1回の生成APIで次を出します。

```text
translated_prompt
translated_chosen
rejected_candidates
translation_quality_score
```

役割は次の通りです。

- `translated_prompt`: 英語promptを自然な日本語会話文脈にしたもの。
- `translated_chosen`: 元の高スコア英語応答を、日本語の理想応答として翻訳・調整したもの。
- `rejected_candidates`: 同じpromptに対する、自然だがchosenより弱い日本語応答候補。

重要なのは、`chosen`はゼロから悪くない応答を作るのではなく、元の高posterior応答を日本語にしたものだという点です。  
一方、`rejected`は同じpromptに対して新しく生成します。

ESConvでは現在、`rejected_candidates` は既定で8件生成します。

## 4. 生成後に再スコアリングする

生成した日本語DPO候補は、そのまま採用しません。

まず `translated_chosen` を、日本語promptに対する応答として再スコアリングします。

```text
日本語prompt + translated_chosen
  -> GPT-5.4系で観測ラベル化
  -> ESConv/回想法ベイズモデルでposterior計算
```

次に、複数の `rejected_candidates` を1つずつ再スコアリングします。

```text
日本語prompt + rejected候補1
日本語prompt + rejected候補2
...
日本語prompt + rejected候補8
  -> それぞれposterior計算
```

その中から、`chosen_posterior - rejected_posterior` が一番大きいrejectedを選びます。

## 5. 採用条件

現在のESConv DPOでは、次の条件をすべて満たす必要があります。

```text
chosen posterior >= 0.70
rejected posterior <= 0.55
score_gap >= 0.25
```

`score_gap` は次です。

```text
score_gap = chosen posterior - rejected posterior
```

例:

```text
accepted score_gap=0.430 chosen=0.969 rejected=0.538
```

この場合は、chosenが十分高く、rejectedが0.55以下で、差も0.25以上あるため採用されます。

## 6. skip high rejected posterior の意味

```text
[STEP 5/6] skip high rejected posterior
```

これは、生成したrejected候補の中で一番良い組み合わせを選んでも、rejectedのposteriorが高すぎたという意味です。

つまり、rejectedがESConv/回想法ベイズモデルから見て「意外と良い応答」になってしまっています。

この場合、DPO学習に入れると次の問題が起きます。

```text
chosen: かなり良い
rejected: これもそこそこ良い
```

このペアでは、モデルに「何を避けるべきか」が伝わりにくくなります。  
そのため、品質を優先してskipします。

## 7. ログの読み方

```text
[STEP 5/6] dpo generation: 105/400 (26.2%) accepted=9 skipped=95 train_000217#5
```

意味は次の通りです。

- `105/400`: 400件の抽出済み候補のうち、105件目まで処理した。
- `26.2%`: このチャンク内の進捗率。
- `accepted=9`: DPO JSONLに採用できた件数。
- `skipped=95`: 条件未満、content filter、JSON不正などで不採用になった件数。
- `train_000217#5`: 元データの会話IDとターン番号。

その直後に出るログが、その候補の結果です。

```text
skip high rejected posterior
```

この場合、105件目はrejectedが高スコアすぎたため不採用です。

```text
accepted score_gap=0.430 chosen=0.969 rejected=0.538
```

この場合、その候補は採用され、DPO JSONLへ追記されます。

## 8. content_filterの場合

生成APIがcontent_filterに引っかかった場合は、すぐには捨てません。

現在の処理では、入力中の固有名・年齢・日付・親密表現などを中立化した安全版でもう一度試します。

```text
content_filter
  -> 入力を安全化して再試行
  -> 成功すれば採用判定へ進む
  -> それでもcontent_filterならskip
```

これは、content_filterに引っかかった候補が必ず低品質とは限らないためです。

## 9. 現在のESConv改善方針

最近のログでは、採用率が低い主因は `skip high rejected posterior` でした。

そのため、品質閾値は緩めず、次の改善を入れています。

```text
rejected候補数: 4 -> 8
ESConv用rejected生成指示を強化
max-output-tokens: 4096 -> 6144
```

狙いは、自然さを保ったまま、ESConv支援応答として明確に弱いrejectedを見つけやすくすることです。

## 10. まとめ

DPO生成は、単に文章を生成して保存しているだけではありません。

```text
高スコア英語応答を選ぶ
  -> 日本語prompt/chosenへ変換
  -> rejected候補を複数生成
  -> chosenを再スコア
  -> rejected候補をすべて再スコア
  -> chosenとrejectedの差が十分大きいものだけ保存
```

そのため、`skip` が多いこと自体は異常ではありません。  
むしろ、DPO学習に入れる比較ペアの品質を守るためのフィルタです。
