# MathDial評価軸 v2

## 位置づけ

`mathdial_oracle_v1`の100 promptを採点した後に評価軸を再検討したため、v1結果を
消したり、v2を同じ標本上の確認的主結果として扱ったりしない。

- v1: 事前固定された元評価。ただし、v2設計に対する探索資料として扱う。
- v2: 軸とrubricを固定した後、v1で未使用のtest qidから新たに選ぶ100 promptで確認する。
- 一般品質: MathDialスタイル得点へ混ぜず、正確性・自然さ・文脈維持の別カテゴリで報告する。
- v2採点後に軸の追加、削除、重み変更は行わない。

「有意差が出る軸」を選ぶのではなく、目的コーパスが定義する教育行動を、一般的な会話品質から
分離して測る。v2でも有意差が出なければ、その結果を保持する。

## 根拠

MathDial原論文は人手評価で、教師応答の`coherence`、`correctness`、
`equitable tutoring`を使用した。equitable tutoringは、学習者が問題と解法空間を考え、
説明し、探索する余地を与えることを指す。また対話全体では、学習者のsolve rateと、
学習者が自力で到達する前に教師が最終解答を伝える`Telling@k`の均衡を測った。

MRBenchはMathDialを含む誤り修正対話を、人手注釈された8軸で評価した。そのうち、
MathDial型の指導過程に直接対応する`mistake identification`、`mistake location`、
`revealing of the answer`、`providing guidance`、`actionability`をv2主軸へ採用した。
`coherence`、`tutor tone`、`human-likeness`は重要だが、目的スタイル以外の一般品質を
強く含むため主合成から分離する。

## 主評価7軸

| axis | 元の評価観点 | v2での扱い | MathDialスタイルとの関係 |
|---|---|---|---|
| `equitable_tutoring` | MathDial人手評価のequitable tutoring | ほぼ直接使用。ただし、質問形や解答非提示だけでは高得点にしない | 学習者に思考・説明・探索の実質的余地を残す |
| `learner_reasoning_diagnosis` | MRBench mistake identification、MathDial correctness | 正答も含むheld-out文脈向けに、正しい・誤り・不完全・混乱の較正へ拡張 | 誤答の誤肯定と正答への誤訂正をともに防ぐ |
| `mistake_location_and_targeting` | MRBench mistake location、Daheim et al. targetedness | 誤りがない場合は未完了点・確認点を対象とする条件付き軸へ拡張 | 最初に修正すべき箇所へFocusする |
| `guidance_quality` | MRBench providing guidance | 正確で関連するhint、説明、例、支援質問を等しく許容 | Probing、Focus、Tellingを内容面から評価する |
| `feedback_actionability` | MRBench actionability | 次に学習者が実行する認知行動を評価 | 自己修正・再計算・説明・検証へつなぐ |
| `answer_revealing_calibration` | MRBench revealing of the answer、MathDial Telling@k | 単純な非提示率ではなく、学習者状態に応じた情報開示の時機を評価 | 早すぎる解答と、必要な説明を避け続ける行動の両方を罰する |
| `teacher_move_stage_alignment` | MathDial Probing / Focus / Telling / Generic taxonomy | taxonomyを会話段階への適合評価として再定義 | 状態に応じてteacher moveを切り替えるBASiS仮説を測る |

## 旧軸からの変更

- `tutoring_style_strength`は複数概念を一つに圧縮していたため主軸から外す。
- `misconception_diagnosis`は、正答に誤りを捏造する失敗も明示的に扱う
  `learner_reasoning_diagnosis`へ置き換える。
- `scaffolding_quality`は、支援内容と次行動を分けて
  `guidance_quality`と`feedback_actionability`にする。
- `premature_answer_avoidance`は、Baseも高得点になる天井効果と、
  Tellingが必要な場面を不当に罰する可能性があるため、条件付きの
  `answer_revealing_calibration`へ置き換える。
- `pedagogical_transition_plausibility`は実際の応答後状態を観測していないため、
  観測したかのような名称をやめる。代わりに、現在状態に対する
  `teacher_move_stage_alignment`と次行動の`feedback_actionability`を測る。
- `natural_japanese`、`overall_quality`等は一般品質として別報告する。

## 確認評価プロトコル

1. v1の100 promptで使用したsample idとqidを除外する。
2. MathDial testからqid・会話単位で一意な100 promptを選ぶ。
3. 元Teacher moveの`probing / focus / telling / generic`で層化する。
4. Base / BASiS-DPO / Random-DPOへ同じprompt、同じ生成条件を使う。
5. モデル名を隠し、応答順をseed固定でランダム化する。
6. 各軸についてFriedman検定を行い、有意な場合だけ対応ありpermutation testと
   Holm補正を行う。
7. Kendall's W、Cohen's dz、rank-biserial、bootstrap 95% CIを報告する。
8. v1とv2を併合して標本数を水増ししない。

## 出典

- Macina et al. (2023), MathDial, Findings of EMNLP:
  https://aclanthology.org/2023.findings-emnlp.372/
- Maurya et al. (2025), Unifying AI Tutor Evaluation / MRBench, NAACL:
  https://aclanthology.org/2025.naacl-long.57/
- Daheim et al. (2024), Stepwise Verification and Remediation, EMNLP:
  https://aclanthology.org/2024.emnlp-main.478/
- Liermann et al. (2024), More Insightful Feedback for Tutoring, EMNLP:
  https://aclanthology.org/2024.emnlp-main.605/
