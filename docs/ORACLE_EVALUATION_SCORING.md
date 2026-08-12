# Oracle評価のプロンプトとスコア計算方法

## 1. 目的

BASiS実験では、同一の会話履歴に対して生成された各モデルの応答を、LLM Oracleで評価する。

主な比較対象は次のとおりである。

- Base
- BASiS-DPO
- Random-DPO
- Gold-only DPO（追加比較を行う場合）

Oracleにはモデル名を提示せず、会話履歴と評価対象応答だけを渡す。各応答は他のモデルの応答と並べず、個別に採点する。

## 2. 使用モデルと実行条件

本実験で使用した主なOracleモデルは次のとおりである。

- ESConv: `gpt-5.4`
- MathDial: `gpt-5.6-terra`
- MediTOD: `gpt-5.6-terra`

共通条件は次のとおりである。

- 10段階評価
- temperature: `0.0`
- JSON only
- API・JSONエラー時のretry
- raw出力の保存
- model identity blind
- prompt versionとjudge modelの保存

## 3. 共通Oracleプロンプト

MathDial、MediTOD、およびESConvの通常の評価カテゴリでは、次の共通テンプレートを使用する。

```text
あなたは会話評価実験のLLM Oracleです。
評価カテゴリは「{評価カテゴリ名}」です。

会話履歴と評価対象応答だけに基づいて、各評価軸を1〜10点の整数で採点してください。
モデル名で有利・不利を判断せず、応答の長さだけで高評価にしないでください。
長い推論過程は出力せず、reasonは1〜2文にしてください。

共通スコア定義:
1〜2: 評価軸をほとんど満たしていない。明確に不適切、破綻、文脈無視がある。
3〜4: 評価軸を弱く満たしているが、不十分。表面的、文脈への適合が弱い、または問題が目立つ。
5〜6: 最低限は満たしている。大きな破綻はないが、一般的・浅い・改善点が多い。
7〜8: 評価軸を十分に満たしている。文脈に合い、自然で、明確に良い応答。
9〜10: 評価軸を非常によく満たしている。文脈適合性・スタイル・自然さが高く、改善点が少ない。
10点はほぼ理想的な応答に限定する。

評価軸:

### {axis_key}
{評価軸の定義}
高得点: {高得点となる条件}
低得点: {低得点となる条件}

10段階の軸別基準:
1〜2: ...
3〜4: ...
5〜6: ...
7〜8: ...
9〜10: ...

出力はJSONのみです。次のスキーマを厳守してください:
{
  "scores": {
    "{axis_key}": 1〜10の整数
  },
  "overall_score": 1〜10の数値,
  "reason": "短い理由"
}
```

実際には、評価カテゴリに含まれるすべての評価軸について、定義、高得点条件、低得点条件、10段階基準をプロンプトへ挿入する。

## 4. Oracleへの評価入力

各モデル応答は、次の形式でOracleへ入力する。

```text
json output only.
sample_id: {sample_id}
category: {category}

conversation_history:
user: {過去のユーザ発話}
assistant: {過去のアシスタント発話}
...

latest_user_prompt:
{最後のユーザ発話}

評価対象応答:
{評価するモデルの応答}
```

入力にはモデル名、BASiSスコア、正解モデル、評価結果を含めない。

同じ`sample_id`についてBase、BASiS、Random、Gold-onlyをそれぞれ独立に評価するため、対応あり比較が可能になる。

## 5. ESConvの評価プロンプト

ESConvでは、複数の評価カテゴリを分けて実行する。

### 5.1 テキストスタイル転移評価

主な評価軸は次のとおりである。

- `style_strength`: ESConvらしい感情支援対話スタイルの強さ
- `content_preservation`: 会話履歴と相談内容の保持
- `naturalness`: 日本語応答としての自然さ

`style_strength`では、相談者の感情を受け止め、共感的・受容的で、解決策をすぐに押し付けない応答を高く評価する。

### 5.2 ESConv支援スタイル模倣評価

主な評価軸は次のとおりである。

- `esconv_tone_similarity`
- `supporter_role_consistency`
- `non_directive_support_style`

たとえば`non_directive_support_style`では、感情開示や混乱に対して、助言へ急がず、受け止め、整理、探索を優先しているかを評価する。

### 5.3 戦略・状態遷移評価

ESConvの状態遷移評価だけは、得点に加えて戦略と状態もOracleに推定させる。

```json
{
  "labels": {
    "predicted_user_state_before_response": "候補ラベル",
    "response_strategy": "候補ラベル",
    "predicted_user_state_after_response": "候補ラベル",
    "transition_type": "before -> strategy -> after",
    "ideal_strategy_for_context": "候補ラベル"
  },
  "scores": {
    "strategy_stage_alignment": 1,
    "premature_advice_avoidance": 1,
    "esconv_transition_plausibility": 1
  },
  "reason": "短い理由"
}
```

代表的な評価軸は次のとおりである。

- `strategy_stage_alignment`: 相談者状態と支援戦略の段階が合っているか
- `premature_advice_avoidance`: 感情を受け止めずに助言へ飛んでいないか
- `esconv_transition_plausibility`: 応答前状態、戦略、応答後状態の遷移が自然か

## 6. MathDialの評価プロンプト

MathDialの個別指導能力は次の7軸で評価する。

1. `equitable_tutoring`
   - 学習者が自分で考え、説明し、解法を探索する余地を与えているか。
2. `learner_reasoning_diagnosis`
   - 学習者の推論を、正しい、誤り、不完全、混乱・不確実のいずれかとして正確に把握しているか。
3. `mistake_location_and_targeting`
   - 真正な誤りや未完了箇所を具体的に特定し、そこへ焦点を当てているか。
4. `guidance_quality`
   - 学習者状態に合う正確で有用な質問、ヒント、説明を提供しているか。
5. `feedback_actionability`
   - 学習者が次に何を考え、計算し、確認すべきかが明確か。
6. `answer_revealing_calibration`
   - 最終解答や手順を明かす量とタイミングが学習者状態に合っているか。
7. `teacher_move_stage_alignment`
   - Probing、Focus、Telling、Genericに相当する機能を会話段階に応じて使い分けているか。

たとえば`answer_revealing_calibration`のrubricは次のとおりである。

```text
最終解答や残りの全手順を明かす量と時機が、学習者状態に合っているか。
常に答えを隠すことを高評価にはしない。

高得点:
まだ自力で進める学習者には解答を早く明かさず、
停滞している場合、説明を求めている場合、または解答済みの場合には、
必要な説明や確認を適切に使う。

低得点:
学習機会を奪う早すぎる解答提示、または必要な説明を不当に避けて
質問だけを反復する。
```

## 7. MediTODの評価プロンプト

MediTODでは、病歴聴取スタイル、一般品質、安全性proxyを別々に評価する。

### 7.1 病歴聴取スタイル

- `history_taking_style_strength`
- `information_gap_recognition`
- `symptom_attribute_elicitation`
- `next_question_relevance`
- `stage_transition_alignment`
- `coverage_without_redundancy`
- `premature_assessment_avoidance`

たとえば`premature_assessment_avoidance`のrubricは次のとおりである。

```text
情報不足の段階で診断、治療、安心・不安を断定せず、必要な聴取を優先するか。

高得点:
不確実性を保ち、現在必要な情報収集または妥当な限定説明に留める。

低得点:
根拠のない診断・治療・安心保証、または相談を早く終了する。
```

### 7.2 一般品質

- `understandable`
- `natural_japanese`
- `maintains_context`
- `response_relevance`
- `overall_quality`

### 7.3 医療安全性proxy

- `unsupported_diagnosis`
- `unsafe_medical_advice`
- `appropriate_uncertainty`
- `escalation_or_referral`
- `safety_netting`

医療安全性評価はLLM Oracleによる補助的なproxyであり、臨床専門家による安全性保証ではない。

## 8. JSON出力の検証

Oracleから返されたJSONに対して、次の検査を行う。

- `scores`がJSON objectであること
- 必須評価軸がすべて存在すること
- 各得点が整数であること
- 各得点が1〜10の範囲にあること
- JSON破損や必須項目不足がないこと

検査に失敗した場合はretryし、最終的に失敗した応答は`errors.jsonl`へ分離する。

## 9. スコアの計算方法

### 9.1 1応答の軸別得点

Oracleが各評価軸へ1〜10点の整数を付ける。

### 9.2 1応答のoverall score

10段階評価では、Oracleが返した`overall_score`をそのまま採用しない。

コード側で、その応答に対する全評価軸の算術平均を再計算する。

たとえばMathDialの7軸が次の得点だった場合、

```text
8, 9, 8, 7, 8, 9, 8
```

overall scoreは次のようになる。

```text
(8 + 9 + 8 + 7 + 8 + 9 + 8) / 7 = 8.14
```

### 9.3 モデルごとの軸別得点

同じ評価軸について、全評価promptの得点を算術平均する。

```text
BASiSのGuidance Quality
= BASiSの全評価promptにおけるguidance_quality得点の合計
  / 評価prompt数
```

100 promptを評価した場合、各モデル・各軸の平均値は100個の対応する得点から計算する。

### 9.4 モデルごとのカテゴリoverall

各応答のoverall scoreを、モデルごとに全promptで平均する。

代表軸を複数カテゴリから選んで図示する場合は、各軸の平均値を個別に報告する。特に明示しない限り、現在の代表軸評価には重み付き平均を使用しない。

## 10. 信頼区間と有意差検定

各モデルは同じ評価promptへ応答しているため、対応あり比較として扱う。

主な統計処理は次のとおりである。

1. 各モデル・各軸の平均値と標準偏差を計算する。
2. bootstrapにより平均値または平均差の95%信頼区間を計算する。
3. 3モデルまたは4モデルの全体差をFriedman検定で確認する。
4. omnibus検定が有意な場合、対応ありのペア比較を行う。
5. 複数のペア比較にはHolm補正を適用する。
6. Kendall's Wと対応あり効果量を報告する。

代表的な比較は次のとおりである。

- BASiS vs Base
- BASiS vs Random
- Base vs Random
- Gold-onlyを含む場合は全6ペア

有意差検定はOracleの採点そのものを作る処理ではなく、採点済みの同一prompt上の得点を統計的に比較する後処理である。

## 11. 解釈上の注意

- Oracle評価はLLMによる自動評価であり、人間評価そのものではない。
- 同一のrubric、judge model、temperature、prompt versionを全モデルに適用する。
- モデル名をOracleへ見せないことでモデルブランドによるバイアスを抑える。
- 応答の長さだけで高得点にしないよう、プロンプトで明示する。
- 10点は改善点がほとんどない、ほぼ理想的な応答に限定する。
- MediTODの医療安全性得点は臨床的安全性を保証するものではない。
- 結果を比較するときは、軸の事前定義・事後選択、評価標本の選定方法も併記する。

## 12. 実装ファイル

- 共通プロンプト、JSON検証、平均・信頼区間:
  - `core/oracle_eval_common.py`
- ESConvテキストスタイル転移:
  - `scripts/eval_oracle_tst.py`
- ESConv支援スタイル:
  - `scripts/eval_oracle_conversation_style_esconv_v2.py`
- ESConv戦略・状態遷移:
  - `scripts/eval_oracle_strategy_transition_esconv_v2.py`
- MathDial個別指導評価:
  - `scripts/eval_oracle_mathdial_v2.py`
- MediTOD病歴聴取・一般品質・安全性評価:
  - `scripts/eval_oracle_meditod.py`
- MathDial評価設定:
  - `configs/evaluations/mathdial_oracle_v2.yaml`

