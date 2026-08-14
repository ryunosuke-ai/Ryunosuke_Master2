# Gemini・Claude WebでのOracle再評価

## 目的

ESConv、MathDial、MediTODの既存3モデル応答を、GPT以外のjudgeでも同じ10段階rubricにより再評価する。

GeminiとClaudeには同じ入力、同じrubric、同じ出力schemaを使用する。違いはjudgeサービスだけとする。

## パケット生成

```bash
python3 -m tools.prepare_cross_model_web_oracle
```

出力先:

```text
artifacts/cross_model_oracle/web_packets_v1/
  esconv/
  mathdial/
  meditod/
```

各データセットには次が生成される。

- `inputs/batch_XXX.jsonl`: Web画面へ添付するblind評価入力
- `prompts/*.txt`: Web画面へ添付する評価プロンプト
- `private_answer_key.jsonl`: 応答IDとBase/BASiS/Randomの対応。judgeへ添付しない
- `manifest.json`: 元ファイルhash、件数、分割条件

## Web画面での実行手順

新しいチャットをカテゴリ・バッチごとに開始する。

1. 評価するカテゴリの`prompts/*.txt`を添付する。
2. `inputs/batch_XXX.jsonl`を1つだけ添付する。
3. 次の短い指示を送る。

```text
添付した評価プロンプトを厳守し、添付JSONLの全行を独立に採点してください。
出力はJSONLのみとし、Markdownや前後の説明を付けないでください。
```

4. 返答をjudge名、データセット、カテゴリ、batch番号が分かる名前で保存する。

例:

```text
results/gemini/esconv/conversation_style/batch_001.jsonl
results/claude/mathdial/pedagogical_v2/batch_001.jsonl
```

## 使用するプロンプト

### ESConv

- `prompts/text_style_transfer.txt`
- `prompts/conversation_style.txt`
- `prompts/strategy_transition.txt`

### MathDial

- `prompts/pedagogical_v2.txt`
- `prompts/general.txt`

### MediTOD

- `prompts/history.txt`
- `prompts/general.txt`
- `prompts/safety.txt`

プロンプトにはGPT Oracleで使った評価軸の定義、高得点・低得点条件、1〜10点の軸別基準をそのまま含める。

## 重要な注意

- `private_answer_key.jsonl`はGemini・Claudeへ添付しない。
- 既存のOracle `raw.jsonl`も、GPTの得点が含まれるため添付しない。
- Webサービス上で会話が継続すると前バッチの判断が影響し得るため、カテゴリ・バッチごとに新しいチャットを使う。
- 同じitemの3モデル応答は別waveへ分離されており、同じbatchには入らない。
- Web版ではAPI版の`temperature=0.0`を固定できない場合がある。使用したサービス、モデル表示名、実行日、設定画面で変更できた項目を記録する。
- GeminiとClaudeの結果を混ぜず、judge別に統計を算出する。
- MediTODの安全性得点はLLMによるproxyであり、臨床的安全性の保証ではない。

