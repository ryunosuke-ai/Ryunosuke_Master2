# MathDial識別力重視追試

## 位置づけ

この追試は、旧MathDial instructionで学習済みのv6 adapterを再利用し、
Teacher moveの選択が必要なheld-out履歴へ評価対象を限定する。

過去の群別結果を確認してから対象群を定義したため、MathDial test全体の無条件な
主評価ではなく、次の仮説を未使用qidで検証する追試として扱う。

> 非Genericで学習者の推論が観測できる場面では、BASiSはBaseおよび
> Random-DPOよりMathDial型の教師戦略を示す。

評価軸は`mathdial_oracle_v2`から変更しない。結果確認後にquota、除外条件、
軸の追加・削除・重みを変更しない。

## 選定条件

- MathDial公式testのみ
- v6とv11で評価済みの200 qidを除外
- qid・conversation単位で一意
- assistant応答後のuser反応が観測可能
- source Teacher moveが`probing / telling / focus`
- 最後のuser発話が20文字以上、または数値・数式を含む8文字以上
- モデル応答とOracle結果は選定に使用しない

150件のquotaと18件の翻訳補欠は
`configs/evaluations/mathdial_discriminative_followup_v1.yaml`で固定する。

## 既知の制約

v6 adapterは学習時にTRLのprompt/completion tokenizer境界Mismatch警告が
発生した成果物である。今回の追試では再学習しないため、この問題は修復されない。
生成promptはv6との整合性を優先し、旧instruction、問題文、直近10発話、末尾`AI:`
を使用する。

## 実行

```bash
SOURCE_RUN=artifacts/mathdial_wildchat/runs/mathdial_wildchat_gpt56_v6_candidates4_mixed \
EXCLUDE_NEUTRAL_RUN=artifacts/mathdial_wildchat/runs/mathdial_wildchat_gpt56_v11_neutral_prompt_v6_length \
RUN_TAG=mathdial_v6_instruction_discriminative_followup_v1 \
EVAL_COUNT=150 \
START_STAGE=prepare_eval \
END_STAGE=report \
WORKERS=4 \
EVAL_CUDA_VISIBLE_DEVICES=0,1 \
PYTHONUNBUFFERED=1 \
./scripts/run_mathdial_instruction_discriminative_v2_watchdog.sh
```

`selection_manifest.json`へ除外qid hash、候補数、quota、補欠利用数を保存する。
主検定とHolm補正は全150件で行い、Teacher move・段階別の表は
`exploratory_descriptive_only`として別出力する。
