# MediTOD評価軸 v1

本評価は、医学知識の正しさと体系的な病歴聴取スタイルを分離する。軸は応答生成前に固定し、結果を見て変更しない。

| axis | definition | source paper | original evaluation dimension | adaptation for MediTOD | reason for inclusion |
|---|---|---|---|---|---|
| history_taking_style_strength | 既知情報を踏まえて病歴を順序立てて聴取する強さ | MediTOD, EMNLP 2024 | CMAS / policy learning | intent・slot列の模倣ではなく、次応答の病歴聴取機能として再定義 | 目的コーパス全体の進め方を直接測る |
| information_gap_recognition | 判断前に不足情報を見つける能力 | Ask Patients with Patience, EMNLP 2025 | Gathering Information | 現在の履歴で未確認の情報を選べるかへ再定義 | BASiSのstate依存選別に対応する |
| symptom_attribute_elicitation | onset、期間、経過、重症度、特徴を引き出す能力 | MediTOD, EMNLP 2024 | slot/attribute annotations | 公式annotationが表す属性確認を自然言語応答のrubricへ変換 | MediTOD固有の細粒度annotationを利用できる |
| next_question_relevance | 直前患者発話に関連する次質問 | Ask Patients with Patience, EMNLP 2025 | Relevant Response Rate | 質問以外の要約・移行も含む次応答関連性へ拡張 | 無関係なテンプレート質問を高評価にしない |
| stage_transition_alignment | 症状から背景歴・要約へ移る時機 | MediTOD, EMNLP 2024 | conversation flow / policy | 会話十分位とslot遷移に沿う段階移行へ再定義 | 単一質問でなく状態遷移を測る |
| coverage_without_redundancy | 聞き漏らしを減らしつつ重複しない | MediTOD, EMNLP 2024; Ask Patients with Patience | slot coverage / information gathering | 1応答で得られる新情報と重複質問を同時に評価 | 質問数の多さを質と誤認しない |
| premature_assessment_avoidance | 情報不足で診断・助言を急がない | MediTODのhistory-taking task設計 | history taking before downstream assessment | 十分な情報がない文脈での早期断定回避へ再定義 | 病歴聴取スタイルと医学知識回答を分離する |

MediTOD: <https://aclanthology.org/2024.emnlp-main.936/>

Ask Patients with Patience: <https://aclanthology.org/2025.emnlp-main.142/>

安全性カテゴリはLLM Oracleによる補助的proxyであり、臨床専門家評価や安全性保証ではない。
