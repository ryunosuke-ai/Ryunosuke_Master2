# DPOパイプラインのログ配置

DPO関連ログは用途別に保存する。現在の発表準備ではESConv支援対話スタイル学習実験を主対象にする。`reminiscence_5000_to_2000` を含むログ名でも、`logs/dpo_pipeline/esconv/` や `logs/oracle_evaluation/esconv/` 配下にあるものはESConv実験ログとして扱う。

```text
logs/dpo_pipeline/
  combined/       旧連続実行や複合ジョブ
  esconv/         現行ESConv DPOパイプライン本体。日付別に `YYYYMMDD/` へ保存する
  random_dpo/     Random-DPO baseline
  reminiscence/   旧回想法DPOパイプライン本体
  legacy/         旧単発DPOスクリプトのログ
```

watchdogログは本体ログと同じ考え方で分ける。

```text
logs/dpo_pipeline_watchdog/
  esconv/         現行ESConv watchdog。日付別に `YYYYMMDD/` へ保存する
  reminiscence/   旧回想法watchdog
```

通常確認する順番:

1. 連続実行なら `logs/dpo_pipeline/combined/`
2. ESConv本体なら `logs/dpo_pipeline/esconv/YYYYMMDD/`
3. Random-DPOなら `logs/dpo_pipeline/random_dpo/YYYYMMDD/`
4. 旧回想法本体なら `logs/dpo_pipeline/reminiscence/`
5. watchdog再起動や停止理由なら `logs/dpo_pipeline_watchdog/esconv/YYYYMMDD/` または `logs/dpo_pipeline_watchdog/reminiscence/`

最新ログを見る例:

```bash
ls -lt logs/dpo_pipeline/combined logs/dpo_pipeline/esconv/* logs/dpo_pipeline/reminiscence | head -n 30
tail -n 120 logs/dpo_pipeline/combined/最新のログファイル名.log
```

ESConvをwatchdog経由で実行した場合は、watchdogが再起動しても同じ本体ログへ追記する。
そのため、通常は1回のwatchdog実行につき次の2ファイルを見る。

```text
logs/dpo_pipeline/esconv/YYYYMMDD/dpo_pipeline_${RUN_TAG}_YYYYMMDD_HHMMSS.log
logs/dpo_pipeline_watchdog/esconv/YYYYMMDD/dpo_pipeline_${RUN_TAG}_watchdog_YYYYMMDD_HHMMSS.log
```

`PIPELINE_LOG_FILE` を明示指定した場合は、ESConv本体ログはそのファイルへ追記する。
