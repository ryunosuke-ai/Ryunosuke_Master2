# MathDial feature prompts v1

## Extraction

```text
You analyze one turn of a one-to-one tutoring dialogue. The target is the
pedagogical process, not mathematical topic similarity. Infer the learner state
before the tutor response, the tutor's single primary strategy, the learner state
after it, and the dialogue stage. Do not reward a response merely for giving a
correct final answer. Distinguish diagnosis, focused guidance, graduated hints,
explanation, and premature telling. If no following learner turn is observed,
use unobserved for the after state. Return JSON only and use exactly one listed
label for each scalar field.

Allowed schema and labels:
{
  "student_state_before": "configs/ontologies/mathdial_v1.yaml#student_states",
  "tutor_strategy": "configs/ontologies/mathdial_v1.yaml#tutor_strategies",
  "student_state_after": "configs/ontologies/mathdial_v1.yaml#student_states",
  "conversation_stage": "configs/ontologies/mathdial_v1.yaml#conversation_stages",
  "style_features": "configs/ontologies/mathdial_v1.yaml#style_features",
  "confidence": "0.0-1.0",
  "short_reason": "one short sentence"
}
```

実際のlabel配列は実行時にversion固定ontologyから展開する。入力には問題文、参照解、
完全履歴、対象Tutor応答、存在する場合の次Student発話だけを入れる。元Teacher moveは入れない。

## Validation

```text
Independently validate a structured tutoring-turn analysis against the transcript.
Check that every label is supported by observable text, that the primary strategy
is the best available label, and that an after-state is not invented without a
following learner turn. Return JSON only with keys valid (boolean),
corrected_extraction (the complete extraction schema), confidence (0-1), and
short_reason.
```

validatorにもTeacher moveを渡さない。Teacher moveは独立した外部一致評価にのみ利用する。
