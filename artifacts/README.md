# artifacts 構成

- `bayes_models/`: LLMで生成したベイズモデル。
- `datasets/`: DPO用の中間生成物と学習データ。
- `evaluations/`: Oracle評価の応答、判定、summary。
- `run_logs/`: チャンク処理の中間成果物とheartbeat。
- `scored_dialogues/`: ベイズモデルでスコアリングした対話候補。
- `training_runs/`: 学習済みLoRAとcheckpoint。

現在の発表準備ではESConv支援対話スタイル学習実験を主対象にします。`reminiscence_5000_to_2000` を含む成果物名の一部は過去のRUN_TAGを引き継いだものですが、実体はESConv実験です。発表用の保持対象は [ESCONV_MANIFEST.md](ESCONV_MANIFEST.md) を参照してください。

`artifacts/` 配下は原則git管理しません。大きなLoRA、Oracle評価結果、DPOデータ、Bayesモデルは発表直前に削除せず、不要判定が明確なキャッシュやheartbeatだけを個別確認して整理します。
