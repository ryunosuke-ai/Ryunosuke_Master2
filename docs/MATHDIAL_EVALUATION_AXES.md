# MathDial評価軸と根拠

評価軸は本評価結果を見る前に`mathdial_oracle_v1`として固定する。

| axis | definition | source paper | original evaluation dimension | adaptation for MathDial | reason for inclusion |
|---|---|---|---|---|---|
| `tutoring_style_strength` | 個別指導者として学習者を導く強さ | Macina et al., Findings of EMNLP 2023; Maurya et al., NAACL 2025 | equitable tutoring; tutor tone; providing guidance | 複数観点をMathDial全体スタイルとして再定義 | 単なる数学解答と個別指導を分離するため |
| `misconception_diagnosis` | 誤り・混乱を認識し焦点を特定する品質 | Maurya et al., NAACL 2025; Daheim et al., EMNLP 2024 | mistake identification/location; targetedness | 直前までの複数ターン履歴を含む診断へ拡張 | 状態に応じた戦略選択の前提だから |
| `scaffolding_quality` | 適量の質問・ヒントで次の一歩を示す品質 | MathDial; Maurya et al., NAACL 2025 | providing guidance; actionability | probing/focusと段階的hintを含む軸へ再定義 | MathDialの主要なteacher moveを測るため |
| `premature_answer_avoidance` | 学習機会を奪う早期解答を避ける度合い | MathDial; Maurya et al., NAACL 2025 | Telling@k; revealing of the answer | 10点を適切な保留、1点を不必要な直接解答とする | MathDial論文の中心的トレードオフだから |
| `pedagogical_transition_plausibility` | learner stateから次状態への指導手の自然さ | MathDial teacher moves by dialogue stage; Wang et al., Findings of EACL 2023 | strategy prediction and dialogue progress | BASiSのstate-strategy-next stateを直接評価する新しい統合軸 | 提案手法固有の遷移仮説を独立に検証するため |
| `teacher_move_alignment` | probing/focus/telling/generic相当機能の適合 | MathDial | teacher move taxonomy | 細粒度ontologyを4上位カテゴリへ写像 | 元ラベルとの外部整合性を評価するため |
| `learner_self_correction_support` | 学習者自身の再検討・修正を促す度合い | MathDial; Maurya et al., NAACL 2025 | solve rate versus telling; active learning/actionability | 単一応答が作る自己修正機会として再定義 | 最終解答提示だけを高評価にしないため |

一般品質の`correctness`, `understandable`, `natural_japanese`, `maintains_context`,
`overall_quality`は目的スタイルと品質低下を分離して報告する。Oracle評価だけで実際の学習効果を
保証したとは主張しない。

主要出典:

- Macina et al. 2023, MathDial, https://aclanthology.org/2023.findings-emnlp.372/
- Maurya et al. 2025, Unifying AI Tutor Evaluation, https://aclanthology.org/2025.naacl-long.57/
- Daheim et al. 2024, Stepwise Verification and Remediation, https://aclanthology.org/2024.emnlp-main.478/
- Wang et al. 2023, Strategize Before Teaching, https://aclanthology.org/2023.findings-eacl.170/
