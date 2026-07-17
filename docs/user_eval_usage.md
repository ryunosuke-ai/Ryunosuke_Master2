# ユーザ評価実施手順

## 1. 評価itemを作成する

作業ディレクトリはリポジトリ直下にする。

```bash
cd /home/ito/Master2
python3 scripts/prepare_user_eval_items.py
```

出力:

- `artifacts/user_eval/items/user_eval_items.jsonl`
- `artifacts/user_eval/items/selection_manifest.json`
- `artifacts/user_eval/items/selected_items.csv`

選定内容だけ確認する場合:

```bash
python3 scripts/prepare_user_eval_items.py --dry-run
```

既定では、Oracle上の `oracle_basis_win` 20件、`oracle_random_win` 5件、`oracle_close` 5件を、seed `20260619` で選定する。

## 2. tmuxでStreamlitを起動する

GPUサーバー上でtmux sessionを作る。

```bash
cd /home/ito/Master2
tmux new -s user_eval
```

tmux内でStreamlitを起動する。

```bash
streamlit run apps/user_eval_app.py --server.address 0.0.0.0 --server.port 8501
```

1人あたり10件や15件に分割したい場合は、Streamlit引数の後に `--` を入れて指定する。

```bash
streamlit run apps/user_eval_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  -- \
  --items-per-participant 15
```

全員に30件を評価してもらう場合は `--items-per-participant` を指定しない。

## 3. tmuxのdetach / attach

Streamlitを起動したままtmuxから抜ける。

```text
Ctrl-b その後 d
```

再接続する。

```bash
tmux attach -t user_eval
```

tmux session一覧を確認する。

```bash
tmux ls
```

## 4. アクセスURLを確認する

サーバー内でIPアドレスを確認する。

```bash
hostname -I
```

または:

```bash
ip -4 addr
```

研究室内LANまたはVPNから、次の形式でアクセスする。

```text
http://<GPUサーバーのIPアドレス>:8501
```

例:

```text
http://192.168.1.10:8501
```

ファイアウォールやVPN設定により外部から見えない場合は、次の代替案を使う。

### SSHポートフォワーディング

手元PCからGPUサーバーへSSHできる場合:

```bash
ssh -L 8501:localhost:8501 <user>@<gpu-server-host>
```

手元PCのブラウザで開く。

```text
http://localhost:8501
```

### ngrok

GPUサーバーでngrokが使える場合:

```bash
ngrok http 8501
```

表示されたHTTPS URLを研究室メンバーに共有する。外部サービスを使うため、回答内容を外部経路に流してよいか事前に確認する。

### Cloudflare Tunnel

Cloudflare Tunnelが使える場合:

```bash
cloudflared tunnel --url http://localhost:8501
```

表示されたURLを共有する。こちらも外部サービス利用の扱いを事前に確認する。

## 5. 評価中の注意

- 評価者は開始画面で「氏名または参加者ID」を入力する。
- 開始画面と各評価画面には、カウンセリング場面での相談支援らしい会話スタイルの説明が表示される。
- 評価者は5つの観点それぞれで、Model A / Model Bのどちらがより当てはまるかを選ぶ。
- 生回答には `participant_name` が保存される。
- 集計CSV、Markdownレポート、発表用グラフには個人名を出力しない。
- 各ブラウザセッションに一意の `session_id` が生成される。
- 同じ氏名または参加者IDで複数回アクセスしても、session_idが異なれば別ファイルに保存される。
- 未回答itemは新規保存され、前へ戻って回答済みitemを修正した場合は同じitemの回答が置換される。

中断・再開:

- 同じブラウザセッションが残っている場合は、そのまま未回答itemから続行できる。
- ブラウザを閉じた場合は、画面に表示された `session_id` を控えていれば開始画面の「再開用session_id」に入力して再開できる。
- `session_id` が分からない場合は新しいセッションとして再開する。この場合も分析時に複数JSONLを統合できる。
- 同じsession_idを複数ブラウザで同時に開く運用は避ける。

前の評価を修正する場合:

- 評価画面の「前へ戻る」を押すと、直前のitemに戻れる。
- 保存済みの評価値とコメントは画面に再表示される。
- 内容を変更して「保存して次へ」を押すと、回答JSONL内の同じitemの行が更新される。
- 分析スクリプトは同じ `participant_id` / `session_id` / `item_id` の回答が複数ある場合でも、最後の回答を採用する。

## 6. 回答データの保存場所

回答JSONLは次に保存される。

```text
artifacts/user_eval/responses/
```

ファイル名:

```text
{participant_id}_{session_id}.jsonl
```

例:

```text
artifacts/user_eval/responses/p_1a2b3c4d5e6f_20260619090000_abcd1234ef567890.jsonl
```

## 7. 回答データをバックアップする

評価中または評価終了後に、回答ディレクトリをタイムスタンプ付きでコピーする。

```bash
mkdir -p artifacts/user_eval/backups
cp -a artifacts/user_eval/responses \
  "artifacts/user_eval/backups/responses_$(date +%Y%m%d_%H%M%S)"
```

圧縮バックアップを作る場合:

```bash
mkdir -p artifacts/user_eval/backups
tar -czf "artifacts/user_eval/backups/responses_$(date +%Y%m%d_%H%M%S).tar.gz" \
  artifacts/user_eval/responses
```

バックアップ先を研究室内の別ストレージへコピーする場合:

```bash
rsync -av artifacts/user_eval/responses/ <backup-host>:/path/to/user_eval_responses/
```

## 8. Streamlitを停止する

tmuxへ戻る。

```bash
tmux attach -t user_eval
```

Streamlitを止める。

```text
Ctrl-c
```

tmux session自体を終了する。

```bash
exit
```

または外からsessionを終了する。

```bash
tmux kill-session -t user_eval
```

## 9. 結果を集計する

回答JSONLをすべて統合して分析する。

```bash
python3 scripts/analyze_user_eval_results.py
```

明示的に入力ファイルを指定する場合:

```bash
python3 scripts/analyze_user_eval_results.py \
  --input artifacts/user_eval/responses/p_xxx_session1.jsonl \
  --input artifacts/user_eval/responses/p_yyy_session2.jsonl
```

出力:

- `artifacts/user_eval/results/normalized_responses.csv`
- `artifacts/user_eval/results/summary.csv`
- `artifacts/user_eval/results/participant_summary.csv`
- `artifacts/user_eval/results/item_summary.csv`
- `artifacts/user_eval/results/report.md`

発表用グラフ:

- `artifacts/user_eval/results/figures/win_rate_bar.png`
- `artifacts/user_eval/results/figures/win_rate_bar.svg`
- `artifacts/user_eval/results/figures/rating_distribution.png`
- `artifacts/user_eval/results/figures/rating_distribution.svg`
- `artifacts/user_eval/results/figures/basis_score_distribution.png`
- `artifacts/user_eval/results/figures/basis_score_distribution.svg`
- `artifacts/user_eval/results/figures/axis_mean_scores.png`
- `artifacts/user_eval/results/figures/axis_mean_scores.svg`

## 10. 発表用に使うファイル

まず確認するファイル:

```bash
cat artifacts/user_eval/results/report.md
```

スライドへ貼る候補:

- `figures/win_rate_bar.png`
- `figures/basis_score_distribution.png`
- `figures/axis_mean_scores.png`
- `summary.csv` のBASiS勝率、Random勝率、平均BASiS基準スコア、95%信頼区間

新しい評価アプリは5軸評価を保存する。分析スクリプトは、各軸のA/B評価をBASiS基準の -2〜+2 に変換し、Oracle評価のweighted overallに対応する重み付き総合スコアを計算する。古い単一 `rating` 形式の回答JSONLも互換のため読み込める。

個人名を含む可能性があるため、`artifacts/user_eval/responses/*.jsonl` は外部公開しない。
