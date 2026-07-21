# ESConv Likertユーザ評価Webアプリ

## 目的

Google Form版と同じ実験指示、20件、A/B分割、匿名化、7評価軸、
最終選択を維持し、比較しやすい画面で回答を収集する。

デスクトップでは上部と左側の会話履歴・応答A/B/Cを固定し、右側の質問列だけを
スクロールして回答する。左側の内容が画面高を超える場合は、左側だけを独立して
スクロールできる。モバイルでは固定を解除し、会話、3応答、質問の順に1列で
表示する。

## 起動

```bash
PORT=8503 ./scripts/run_esconv_likert_user_eval_web.sh
```

既定では`0.0.0.0:8503`で待ち受ける。同じ研究室ネットワークから、
サーバのIPアドレスを使って実験A/BそれぞれのURLへアクセスする。

```text
実験A: http://<サーバのIPアドレス>:8503/?experiment=A
実験B: http://<サーバのIPアドレス>:8503/?experiment=B
```

サーバは1プロセスだけ起動すればよい。URLの`experiment`指定によって、
同じ回答DB内で実験A/Bを分離する。同じ氏名を異なる実験へ重複登録することは
できないため、研究担当者が割り当てた側のURLだけを参加者へ送る。

サーバIPは次で確認できる。

```bash
hostname -I
```

ポートや保存先は環境変数で変更できる。

```bash
HOST=0.0.0.0 \
PORT=8503 \
DATABASE=artifacts/user_eval/web/esconv_likert_responses.sqlite3 \
./scripts/run_esconv_likert_user_eval_web.sh
```

起動時に実験A/Bの配布用URLが端末へ表示される。複数のネットワークがある場合は、
`PUBLIC_HOST=192.168.1.17`のように参加者から到達できるIPを明示する。

## 常時稼働

短期間ならtmuxで実行する。

```bash
tmux new -s esconv-survey
PORT=8503 ./scripts/run_esconv_likert_user_eval_web.sh
```

`Ctrl+B`、続けて`D`でdetachする。再接続は次を使う。

```bash
tmux attach -t esconv-survey
```

OS再起動後も自動起動する場合は、
`deploy/esconv-likert-survey.service.example`を環境に合わせて編集し、
systemdへ登録する。登録には管理者権限が必要になる。

## 回答の流れ

1. 実験指示、3つの特徴、良い・良くない応答例を読む。
2. 氏名を入力し、研究目的での保存・利用へ同意する。
3. 配布されたURLに応じて実験AまたはBへ固定して割り当てられる。
4. 10件を1件ずつ評価する。
5. 各itemを保存するとSQLiteへ即時反映される。
6. 同じ氏名で再度開くと、同じ実験の未完了itemから再開する。

公開JSONLだけを画面へ読み込む。`private_model_mapping.jsonl`と
`answer_key_private.csv`はWebアプリから読み込まず、参加者へ送らない。

## 回答保存

既定DB:

```text
artifacts/user_eval/web/esconv_likert_responses.sqlite3
```

SQLiteはWALと30秒のbusy timeoutを使い、研究室内の同時回答を扱う。
DBには氏名が含まれるため、Gitへ追加せず、研究担当者だけがアクセスする。

研究者用のlong形式CSVは次で出力する。

```bash
python3 -m tools.export_esconv_likert_web_responses \
  --database artifacts/user_eval/web/esconv_likert_responses.sqlite3 \
  --output artifacts/user_eval/web/esconv_likert_responses_long.csv
```

CSVも氏名を含むため、参加者や公開リポジトリへ共有しない。

## 運用上の注意

- 研究室LANまたはVPN内で使う。
- インターネットへ直接公開する場合は、HTTPSとアクセス認証を追加する。
- 実験開始後は公開item、匿名配置、質問文を変更しない。
- DBを定期的にバックアップする。
- 事前に研究室内の1〜2名で画面幅、所要時間、保存・再開をpilot確認する。
