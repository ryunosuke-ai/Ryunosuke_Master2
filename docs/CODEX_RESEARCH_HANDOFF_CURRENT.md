# Codex研究引き継ぎメモ

このファイルは、新しいCodex会話を始めたときに最初に読むための現状メモです。
2026-06-09時点の主実験は、回想法ではなく **ESConvベースの支援対話実験** です。

## 現在の研究目的

小規模高品質コーパスであるESConvから、会話状態・会話戦略・状態遷移をLLMで分析し、ベイズ対話モデルを作る。
そのベイズモデルで大規模対話候補を評価し、ESConvらしい支援応答を選んでDPO preferenceデータを作成する。
最後に `Qwen/Qwen3.5-27B` をLoRA/DPO学習し、DPO後モデルがESConvらしい支援応答を再現できるかを検証する。

処理の中心は次の流れ。

```text
ESConv小コーパス
  -> GPT-5.4-proで会話状態・会話戦略・状態遷移を分析
  -> ESConv transition Bayes modelを生成
  -> DailyDialog候補をGPT-5.4で観測ラベル評価
  -> posteriorで高スコア応答を抽出
  -> DailyDialog DPO + ESConv gold DPOを混合
  -> Qwen3.5-27BをLoRA/DPO学習
  -> base QwenとDPO後QwenをOracle評価で比較
```

注意: `RUN_TAG=reminiscence_5000_to_2000` や成果物名の `reminiscence` は、現在の実験名として残っているだけ。中身はESConvベースの実験であり、回想法実験ではない。

## 最重要方針

- 現在はESConv実験を主軸にする。古い回想法中心の説明や評価方針は参照しすぎない。
- 研究として見るべき点は、単なる総合応答品質ではなく、ESConv由来の支援戦略がDPO後モデルに反映されたか。
- DPO学習データの品質、Oracle評価軸、base/DPOへ渡すlocal prompt条件を明示して比較する。
- モデル名は環境変数で差し替え可能にする。既定は分析 `gpt-5.4-pro`、大量評価 `gpt-5.4`、ローカル学習対象 `Qwen/Qwen3.5-27B`。

## 完了済みのDPO学習

次のLoRA/DPO学習は完了済み。

```text
artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit
```

確認ログ:

```text
logs/dpo_pipeline/esconv/20260608/dpo_pipeline_reminiscence_5000_to_2000_20260608_122134.log
```

ログ上の完了情報:

- `train_loss=0.3111`
- `epoch=1`
- LoRA adapter保存成功
- `ESConv DPO chunked pipeline completed`
- 完了時刻: 2026-06-08 14:09:16 JST

学習時の主要出力:

```text
bayes_model: artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json
dailydialog_dpo_data: artifacts/datasets/dailydialog_ja_dpo_preferences_reminiscence_5000_to_2000_daily.jsonl
esconv_gold_dpo_data: artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl
dpo_data: artifacts/datasets/esconv_mixed_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl
training_output: artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit
```

## DPOデータ

学習に使った混合DPOデータ:

```text
artifacts/datasets/esconv_mixed_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl
```

実ファイル確認結果:

- 総件数: 2500件
- DailyDialog由来: 2000件
- ESConv gold DPO: 500件
- chosen posterior平均: 0.9620
- score gap平均: 0.3871
- `acceptance_rule`: `strict` 506件、`gap_rescue` 1983件、未設定 11件

ESConv gold側のstrategy内訳:

- `Restatement or Paraphrasing`: 68
- `Question`: 68
- `Affirmation and Reassurance`: 67
- `Others`: 67
- `Providing Suggestions`: 66
- `Reflection of feelings`: 65
- `Information`: 59
- `Self-disclosure`: 40

関連ファイル:

```text
artifacts/datasets/dailydialog_ja_dpo_preferences_reminiscence_5000_to_2000_daily.jsonl
artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl
artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.manifest.json
artifacts/run_logs/reminiscence_5000_to_2000/chunks/
```

## v2 Oracle評価結果

既存のESConv v2 Oracle評価結果:

```text
artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v2/summary.json
artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v2/responses.jsonl
artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v2/judgments.jsonl
```

実ファイル確認結果:

- 評価件数: 100
- `base_mean`: 86.28
- `dpo_mean`: 81.23
- `gap`: -5.05
- `dpo_win_rate`: 30.00%
- base wins: 70
- dpo wins: 30
- ties: 0

総合ではbaseが上回った。
ただし、カテゴリ別では `suggestion_timing` のみDPOがbaseを上回った。

- `suggestion_timing` base: 84.6
- `suggestion_timing` dpo: 86.1
- gap: +1.5
- dpo win rate: 60.00%

解釈:
v2では質問、問題探索、具体的な次の一歩などの汎用支援品質が強く効き、ESConvらしさそのものを十分に分離できていない可能性がある。
一方でDPO後モデルは「早すぎる助言を避ける」「まず受け止める」というESConv的傾向を一部学習できている可能性がある。

## v3 Oracle評価

v2の反省を受けて、ESConvらしさを明示的に測るv3評価を追加済み。

追加・更新された主なファイル:

```text
configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl
scripts/run_oracle_evaluation_esconv_v3_strategy.sh
tools/run_oracle_evaluation.py
tests/test_oracle_evaluation.py
```

`tools/run_oracle_evaluation.py` には `esconv_strategy_v3` presetがある。
v3評価promptは100件あり、`configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl` で確認できる。

v3の評価軸:

- `esconv_strategy_adherence`
- `emotional_reflection_validation`
- `premature_advice_avoidance`
- `supportive_tone`
- `contextual_grounding`
- `conversational_progression`
- `overall_helpfulness`

主指標:

- 最重要: `esconv_core_score`
- 重要: `esconv_strategy_adherence`, `emotional_reflection_validation`, `premature_advice_avoidance`
- 補助: `weighted_esconv_overall`, `supportive_tone`, `contextual_grounding`
- 弱点分析: `conversational_progression`, `overall_helpfulness`

`esconv_core_score` は、今回の研究目的である「小規模ESConvコーパス由来の会話戦略がDPO後モデルに反映されたか」を見る主指標。
重みは以下。

```text
esconv_strategy_adherence: 0.40
emotional_reflection_validation: 0.35
premature_advice_avoidance: 0.25
```

`weighted_esconv_overall` は補助指標。
ESConvらしさを重視しつつ、支援応答全体の品質も少し含む。

```text
esconv_strategy_adherence: 0.25
emotional_reflection_validation: 0.25
premature_advice_avoidance: 0.20
supportive_tone: 0.10
contextual_grounding: 0.10
conversational_progression: 0.05
overall_helpfulness: 0.05
```

## v3 Oracle評価の現在状況

2026-06-09時点で、v3 Oracle評価は実行開始済みだが、完了成果物はまだ出ていない。

想定出力先:

```text
artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy
```

現状確認:

- 出力ディレクトリは存在する。
- ただし `summary.json`, `responses.jsonl`, `judgments.jsonl` はまだ存在しない。
- 実行ログは `logs/oracle_evaluation/esconv/oracle_eval_v3_strategy_reminiscence_5000_to_2000_20260609_111046.log`。
- ログ上ではlocal generation 100/100まで完了。
- Oracle判定は `oracle completed 51/100 esconv_v3_051` 付近でログが終わっている。
- 現在動いている `run_oracle_evaluation` プロセスは確認できなかった。

したがって、次に新しいCodexで見るべきことは、v3評価が途中停止した原因確認、または同じ設定での再実行。

想定実行コマンド:

```bash
RUN_TAG=reminiscence_5000_to_2000 \
PROMPTS=configs/evaluation_prompts/esconv_oracle_eval_v3_strategy_100.jsonl \
SMALL_CORPUS=data/esconv_analysis_corpus_reminiscence_5000_to_2000.jsonl \
BAYES_MODEL=artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json \
DPO_COMPARE_LORA_PATH=artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit \
OUTPUT_DIR=artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy \
ORACLE_MODEL=gpt-5.4-pro \
ORACLE_WORKERS=2 \
./scripts/run_oracle_evaluation_esconv_v3_strategy.sh
```

再実行後は、まず次を確認する。

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/summary.json")
data = json.loads(p.read_text())
for key in [
    "esconv_core_score",
    "weighted_esconv_overall",
    "axis_scores",
    "dpo_win_rate",
    "base_win_rate",
    "tie_rate",
]:
    print(key, json.dumps(data.get(key), ensure_ascii=False, indent=2))
PY
```

## local promptの扱い

現在の評価では、DPO学習時と同じ `instruction` 形式のpromptをbase/DPO両方に渡している。

実装:

```text
core/dpo_prompting.py
tools/run_oracle_evaluation.py --local-prompt-mode instruction
```

`core/dpo_prompting.py` の `INSTRUCTION_LINES` 方針:

- 次のAI返答を生成
- 日本語で1から2文
- 共感や具体語の拾いを使う
- 必要な時だけ質問を1つ添える

これは学習条件と評価条件を揃える意味では妥当。
一方で、baseにも同じ支援的instructionが与えられるため、DPOとの差が小さくなる可能性がある。

v3でも差が小さい場合の補助実験候補:

- `LOCAL_PROMPT_MODE=context_only` でv3評価を再実行し、instructionによるbase底上げを弱めた比較を見る。
- よりESConv方略選択を明示した学習promptで再学習する。
- v3の `esconv_core_score` が改善しているが `overall_helpfulness` が落ちている場合は、ESConvらしさと汎用支援品質のトレードオフとして分析する。

## 主要スクリプト

ESConv DPOパイプライン:

```text
scripts/run_esconv_dpo_then_oracle.sh
scripts/run_dpo_pipeline_esconv_2000_watchdog.sh
scripts/run_dpo_pipeline_esconv_2000_chunked.sh
scripts/run_esconv_then_reminiscence_tail.sh
```

Oracle評価:

```text
scripts/run_oracle_evaluation_esconv.sh
scripts/run_oracle_evaluation_esconv_v3_strategy.sh
tools/run_oracle_evaluation.py
```

ESConvデータ・DPO生成:

```text
tools/prepare_esconv_for_analysis.py
tools/analyze_esconv_corpus_transition_bayes.py
tools/build_esconv_gold_dpo.py
tools/translate_and_generate_dpo.py
tools/stream_dpo_from_scored.py
```

prompt共通化:

```text
core/dpo_prompting.py
```

## 主要成果物

ベイズモデル:

```text
artifacts/bayes_models/generated_transition_bayes_model_esconv_reminiscence_5000_to_2000.json
```

DPOデータ:

```text
artifacts/datasets/esconv_mixed_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl
artifacts/datasets/dailydialog_ja_dpo_preferences_reminiscence_5000_to_2000_daily.jsonl
artifacts/datasets/esconv_gold_ja_dpo_preferences_reminiscence_5000_to_2000.jsonl
```

学習済みLoRA:

```text
artifacts/training_runs/qwen35_bayes_dpo_lora_reminiscence_5000_to_2000_ep1_lr5e-6_r8_a16_no4bit
```

v2 Oracle評価:

```text
artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v2/
```

v3 Oracle評価予定:

```text
artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy/
```

## 新しいCodexセッションで最初に確認すること

1. `docs/CODEX_RESEARCH_HANDOFF_CURRENT.md` を読む。
2. `git status --short` で未コミット変更を確認する。
3. v3評価が完了しているか確認する。

```bash
ls -la artifacts/evaluations/oracle_eval_runs/reminiscence_5000_to_2000_oracle_esconv_v3_strategy
tail -n 80 logs/oracle_evaluation/esconv/oracle_eval_v3_strategy_reminiscence_5000_to_2000_20260609_111046.log
```

4. `summary.json` ができていれば、最初に `esconv_core_score` を見る。
5. `summary.json` がなければ、v3評価の再実行または途中再開方針を決める。

## テスト

軽量テストはAPIやGPUを呼ばない形で実行する。

```bash
python3 -B -m pytest -p no:cacheprovider -v
```

v3 Oracle評価まわりを狭く確認するなら次を使う。

```bash
python3 -B -m pytest -p no:cacheprovider -v tests/test_oracle_evaluation.py
```

GPUがない環境ではQwenの実学習は行わず、学習スクリプトは `--dry-run` でデータ形式だけ確認する。

## commit / push 運用

現在のpush先は新リポジトリの `origin`。

```text
origin: git@github.com:ryunosuke-ai/Ryunosuke_Master2.git
old-origin: git@github.com:ryunosuke-ai/Ryunosuke_Master.git
```

作業後は、必要な軽量テストを実行し、commitし、新しい `origin` へpushする。push前には必ず次を確認する。

```bash
git status --short
git diff --stat
git diff --cached --stat
```

`git add .` は使わず、stageは明示ファイルのみ行う。force pushは禁止。push拒否や認証エラーが出た場合は停止して報告する。

## コミットしないもの

`.env`、APIキー、個人情報、生ログ、大容量データ、学習済みモデル、`data/`、`logs/`、`artifacts/` 配下の生成物は原則コミットしない。

特に `logs/`, `artifacts/run_logs/`, `artifacts/training_runs/`, `artifacts/datasets/`, `artifacts/evaluations/`, `artifacts/bayes_models/`, `hf_cache/`, 生JSONLはpush禁止。発表用評価結果は、完全成果物ではなく軽量な `summary.json` と `manifest.json` だけを `docs/results/` へコピーしてGit管理する。

コミット対象にしやすいもの:

```text
core/
tools/
scripts/
configs/evaluation_prompts/
tests/
docs/
README.md
AGENTS.md
```
