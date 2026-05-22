# Step 5 - LLM/NLI Judge Faithfulness Evaluation

## Goal

The baseline QA evaluation already includes a lexical faithfulness proxy. To add a stronger semantic check, an optional judge-based evaluator was implemented in `scripts/evaluate_llm_judge.py`.

The evaluator supports three modes:

- `nli`: local multilingual NLI judge model
- `local-flan`: local FLAN-T5 text-to-text judge
- `openai`: optional API-based LLM judge if `OPENAI_API_KEY` is available

The final reported run uses the local multilingual NLI judge because no API key was available and FLAN-T5-small was not reliable enough for Turkish legal text.

## Method

Input:

```bash
outputs/qa_eval_extractive_bm25_full.json
```

Command:

```bash
python scripts/evaluate_llm_judge.py --input outputs/qa_eval_extractive_bm25_full.json --provider nli --output outputs/nli_judge_faithfulness_full.json
```

Model:

```text
MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli
```

For each QA row, the judge receives the answer and the top retrieved source text used by the extractive generator. The answer is counted as faithful if the NLI model predicts entailment with probability at least 0.5.

## Result

| Metric | Value |
|---|---:|
| Evaluated QA examples | 240 |
| Judge-supported answers | 206 |
| Judge-rejected answers | 34 |
| Judge faithfulness score | 0.858 |

## Interpretation

The judge-based faithfulness score is lower than the lexical proxy score of 0.961, which is expected because the NLI judge is stricter and checks semantic entailment rather than token overlap. The score still indicates that most answers are supported by the retrieved source text.

This result strengthens the grounding evaluation section of the report: the system is not only retrieving gold sources frequently, but most generated answers are also semantically supported by their retrieved source according to an independent multilingual judge model.
