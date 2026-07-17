# Top-Conference Oracle 評価軸メモ v2

このメモは、10段階Oracle評価で使う評価軸を、論文で説明しやすい形に整理したものです。

v2では、`conversation_style` と `strategy_transition` を再定義しました。目的は、一般的な会話品質や会話継続性ではなく、ESConvらしい支援スタイル・支援過程を評価することです。

## 評価カテゴリの役割

| category | 役割 | 論文での位置づけ |
| --- | --- | --- |
| `conversation_style_esconv_v2` | ESConv支援者らしい会話スタイルを評価する | 主結果 |
| `strategy_transition_esconv_v2` | ESConvらしい支援過程として自然な戦略・状態遷移かを評価する | 主結果 |
| `tst` | Text Style Transferとして、ESConv支援スタイルへの転移を評価する | 主結果 |
| `usr_quality` | 一般的な対話品質を評価する | 欠点・トレードオフ分析 |

元の `conversation_style.engagingness` や `usr_quality.interesting_or_engaging` は、相談者が次に話しやすいか、会話が広がるかを強く見る軸です。これは一般対話品質として重要ですが、目的コーパスらしさそのものとは分けて扱います。

## 論文で使う代表5軸

主結果としては、以下の5軸を代表的な評価軸として使うのが最も自然です。

| 軸 | 評価するもの |
| --- | --- |
| `tst.style_strength` | ESConvらしい感情支援スタイルの強さ |
| `conversation_style_esconv_v2.esconv_tone_similarity` | ESConv支援者らしい共感的・受容的トーンへの近さ |
| `conversation_style_esconv_v2.supporter_role_consistency` | 支援者としての役割・態度の一貫性 |
| `conversation_style_esconv_v2.non_directive_support_style` | 助言へ急がず、受容・整理・探索を優先する支援スタイル |
| `strategy_transition_esconv_v2.premature_advice_avoidance` | 感情開示に対して、共感・受容を挟まず助言へ飛んでいないか |

この5軸は、単にスコアが良かった軸ではなく、研究目的である「ESConvらしい支援対話スタイルの模倣」に直接対応しています。

## 欠点として示す軸

提案手法の弱点として、以下の軸を別に示します。

| 軸 | 解釈 |
| --- | --- |
| `conversation_style.engagingness` | 相談者が次に話しやすいか、会話が広がるか |
| `usr_quality.interesting_or_engaging` | 一般対話として興味深く、返信したくなるか |
| `usr_quality.overall_quality` | 一般対話品質として総合的に良いか |

これらは、ESConvらしさではなく、一般的な会話継続性やユーザ体験に近い軸です。BASiS / Bayes-DPO はこれらで弱く出るため、論文では「目的コーパスらしさの獲得と一般会話品質の間にトレードオフがある」と説明できます。

## 1. conversation_style_esconv_v2

`conversation_style_esconv_v2` は、ESConv支援者らしい会話スタイルを評価するカテゴリです。

元の `conversation_style` では `engagingness` が含まれていましたが、これは一般会話品質や会話継続性に近いため、v2では主スコアから外しました。

### esconv_tone_similarity

応答のトーンが、ESConv支援者らしい共感的・受容的・非断定的な話し方に近いかを評価します。

高評価になる応答:

- 相談者の感情を丁寧に受け止めている
- 穏やかで非評価的
- 断定や押しつけが少ない
- 一般雑談ではなく、相談支援のトーンになっている

低評価になる応答:

- 一般雑談調
- 事務的
- 説教的
- 断定的
- 感情を十分に受け止めない

この軸は、ESConvらしい「声色」や「支援者としての温度」を見る軸です。

### supporter_role_consistency

会話履歴の中で、ESConvの支援者としての役割・態度を一貫して保てているかを評価します。

高評価になる応答:

- 支援者としての態度がぶれない
- 共感、受容、探索、整理を自然に使う
- 履歴の支援的な流れと合っている
- 相談者を評価したり急かしたりしない

低評価になる応答:

- 急に雑談相手のようになる
- 説教者のようになる
- 問題解決者として助言だけを急ぐ
- 相談者の感情よりも正論を優先する

この軸は、単発の自然さではなく、会話全体の中で支援者役割を保てているかを見ます。

### non_directive_support_style

感情開示や混乱に対して、助言へ急がず、受け止め・整理・探索を優先するESConvらしい支援スタイルかを評価します。

高評価になる応答:

- 強い感情にはまず共感・受容で応答する
- すぐに解決策を出さない
- 相談者の状態を整理する
- 必要に応じて穏やかに探索する

低評価になる応答:

- 感情受容を挟まず助言する
- 行動指示を急ぐ
- 一般論で片付ける
- 相談者の感情段階を飛ばす

この軸は、BASiS / Bayes-DPO の狙いにかなり近い軸です。ESConvらしい支援では、特に感情開示に対してすぐ助言へ飛ばないことが重要です。

## 2. strategy_transition_esconv_v2

`strategy_transition_esconv_v2` は、応答の戦略と状態遷移が、ESConvらしい支援過程として自然かを評価するカテゴリです。

元の `strategy_transition` は「会話が自然に進むか」「相談者が次に話しやすいか」という一般的な会話進行の良さを含んでいました。v2では、一般的な会話継続性ではなく、ESConvの支援プロセスとして妥当かを見るように再定義しました。

このカテゴリでは、Oracleが以下のラベルも出します。

- 応答前の相談者状態
- 応答戦略
- 応答後の相談者状態
- その文脈で理想的だった戦略

例:

```text
emotional_disclosure -> empathy_validation -> feeling_organized
```

これは、相談者が感情を吐露している状態に対して、共感・妥当化を行い、その結果として少し整理された状態へ進む、という支援過程を表します。

### strategy_stage_alignment

相談者状態に対して、ESConvでよく見られる支援戦略の段階に合っているかを評価します。

高評価になる応答:

- 感情開示には共感・受容を返す
- 混乱には整理を促す
- 状況説明には探索を行う
- 解決検討の段階では助言や情報提供を行う

低評価になる応答:

- 感情開示に対してすぐ助言する
- まだ整理できていない相談者に行動提案を急ぐ
- 探索すべき場面で一般論だけ返す
- 状態に対して戦略が早すぎる、遅すぎる、ずれている

この軸は、応答戦略がESConvの支援段階に合っているかを見る軸です。

### premature_advice_avoidance

感情開示・混乱・強い不安に対して、共感・受容・探索を挟まず助言へ飛んでいないかを評価します。

高評価になる応答:

- 強い感情をまず受け止める
- 助言や行動提案を急がない
- 相談者の感情段階に沿っている
- 必要に応じて探索や整理を挟む

低評価になる応答:

- つらさを受け止めずに解決策を出す
- 一般論や指示が前に出る
- 相談者の感情よりも対処法を優先する
- ESConvの支援過程として段階を飛ばしている

この軸は、v2のstrategy系で最も代表的な軸です。論文で主張する「ESConvらしい支援過程」に直結します。

### esconv_transition_plausibility

応答前状態、応答戦略、応答後状態の遷移が、ESConvらしい支援過程として自然かを評価します。

高評価になる応答:

- `before_state -> strategy -> after_state` が支援過程として妥当
- 感情受容から整理へ自然につながる
- 探索から解決検討へ無理なく進む
- ESConvらしい状態変化になっている

低評価になる応答:

- 一般的には会話が続いても、支援過程としては唐突
- 感情受容や整理を飛ばしている
- 状態変化が不自然
- 応答後状態の見立てが文脈に合わない

この軸は、単なる「会話の滑らかさ」ではなく、ESConv支援過程としての遷移の妥当性を見ます。

## 3. tst

`tst` は `Text Style Transfer` の略です。元の文脈を保ったまま、ESConvらしい支援スタイルに寄せられているかを評価します。

v2でも、このカテゴリは変更しません。すでにBASiS / Bayes-DPO の強みが安定して出ており、Text Style Transferの評価として妥当だからです。

### style_strength

ESConvらしい感情支援スタイルがどれくらい強く出ているかを評価します。

高評価になる応答:

- 感情を受け止める
- 共感的・受容的
- 助言に急がない
- 温かい支援トーンがある

低評価になる応答:

- 事務的
- 冷たい
- 一般雑談的
- 感情を無視する
- すぐ解決策を押し付ける

代表5軸の1つです。提案手法の狙いに最も近い評価軸です。

### content_preservation

会話履歴やユーザ発話の内容を保てているかを評価します。

高評価になる応答:

- ユーザの悩みを正しく踏まえている
- 感情や状況を誤解していない
- 話題を逸らさない
- 余計な内容を足しすぎない

低評価になる応答:

- 文脈と関係ない
- 悩みを誤解している
- 一般論だけで返す
- ユーザが言っていないことを勝手に追加する

スタイルだけ良くても、元の内容を壊していたら低評価になります。

### naturalness

応答が自然で読みやすいかを評価します。

高評価になる応答:

- 文法的に自然
- 会話として違和感がない
- 過度に冗長ではない
- テンプレート感が少ない

低評価になる応答:

- 不自然な表現
- 文の破綻
- 機械的
- 硬すぎる

TSTでは、`style_strength` が支援スタイルの強さ、`content_preservation` が元内容の保持、`naturalness` が自然な文としての品質を見ます。

## 4. usr_quality

`usr_quality` は、一般的な対話品質を評価するカテゴリです。

このカテゴリは、目的コーパスらしさではなく、普通に良い対話応答かを見るためのものです。BASiS / Bayes-DPO の弱点やトレードオフを示すために使います。

### understandable

意味が通っていて理解可能かを評価します。

### natural

人間の会話文として自然かを評価します。

### maintains_context

文脈を維持しているかを評価します。

### interesting_or_engaging

応答が興味深く、次に返信したくなるかを評価します。

この軸は、BASiS / Bayes-DPO が弱く出やすい軸です。ESConvらしい受容的応答はできていても、問いかけや会話を広げる要素が少ない場合、この軸で低く評価されます。

### overall_quality

一般的な対話応答として総合的に良いかを評価します。

この軸も、BASiS / Bayes-DPO がBaseやRandom-DPOを下回ることがあります。論文では、目的スタイル模倣と一般会話品質のトレードオフとして説明します。

## 結果の解釈

v2評価では、`conversation_style_esconv_v2` でBASiS / Bayes-DPO がBaseとRandom-DPOの両方を有意に上回りました。特に、ESConv支援者らしいトーン、支援者役割の一貫性、助言に急がない非指示的支援スタイルで改善が見られました。

`strategy_transition_esconv_v2` はカテゴリ全体では有意差に届きませんでしたが、`premature_advice_avoidance` ではBASiS / Bayes-DPO がBaseとRandom-DPOの両方を有意に上回りました。これは、提案手法が感情開示に対して助言へ急がず、ESConvらしい支援過程を再現しやすくなっていることを示します。

一方で、`engagingness` や `usr_quality.interesting_or_engaging` ではBASiS / Bayes-DPO が低く出ています。これは、提案手法がESConvらしい受容的・非指示的スタイルを強める一方で、一般的な会話の広がりや返信しやすさを弱める可能性があることを示します。

論文では、次のように整理するとよいです。

```text
BASiS improves ESConv-specific supportive style and support-process alignment,
especially in style strength, ESConv tone similarity, supporter role consistency,
non-directive support style, and premature advice avoidance.
However, BASiS performs worse on general engagingness and USR quality,
suggesting a trade-off between corpus-specific supportive style imitation
and general conversational engagement.
```

日本語では、次のように書けます。

```text
提案手法は、ESConvらしい支援スタイルを表す代表的な5軸において改善を示した。
特に、感情を受け止めるトーン、支援者役割の一貫性、助言に急がない非指示的支援過程で改善が見られた。
一方で、engagingness や USR quality では低下しており、
目的コーパスらしさの獲得と一般的な会話継続性の間にトレードオフがあることが示唆される。
```
