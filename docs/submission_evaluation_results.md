# Submission Evaluation Results

This file records the latest checks aligned with the instructor submission notes.

## 1. Custom Data Support

Verified commands:

```bash
python scripts/validate_custom_data.py --data-dir sample_custom_data --require-benchmark
python scripts/evaluate_qa.py --data-dir sample_custom_data --retriever bm25 --generation-mode extractive --top-k 3 --output outputs/sample_custom_qa.json
python scripts/run_base_vs_finetuned_eval.py --data-dir sample_custom_data --output-dir outputs/sample_submission_eval --top-k 3
python scripts/demo_cli.py --data-dir sample_custom_data --question "Kasten oldurme sucu nedir?"
```

Result: custom corpus, custom benchmark, CLI demo, and base-vs-fine-tuned wrapper all ran successfully on `sample_custom_data/`.

## 2. Base RAG vs Fine-Tuned Reranker RAG

Same corpus, same benchmark, same answer generator:

```text
data/ corpus
eval_qa_150.jsonl benchmark, 150 questions
extractive answer generator
```

Base RAG:

```text
BM25 retrieval -> extractive answer
```

Fine-tuned RAG:

```text
BM25 top-15 candidates -> fine-tuned cross-encoder reranker -> top-5 -> extractive answer
```

Command:

```bash
python scripts/run_base_vs_finetuned_eval.py \
  --data-dir data \
  --output-dir outputs/submission_eval_bm25_vs_finetuned_reranker \
  --base-retriever bm25 \
  --finetuned-retriever bm25 \
  --finetuned-reranker-model C:\Users\bulent\Documents\Codex\2026-05-21\NLP-Term-Project-push\outputs\models\legal_cross_encoder_reranker_full_cpu_128 \
  --rerank-top-n 15 \
  --top-k 5
```

| Metric | Base BM25 RAG | Fine-tuned reranker RAG | Delta |
|---|---:|---:|---:|
| Token F1 | 0.7030 | 0.6313 | -0.0717 |
| ROUGE-L | 0.7016 | 0.6214 | -0.0802 |
| Top-1 source hit | 0.9600 | 0.6200 | -0.3400 |
| Top-5 source hit | 0.9933 | 0.9000 | -0.0933 |
| Citation accuracy | 0.9600 | 0.6200 | -0.3400 |
| Faithfulness proxy | 0.9282 | 0.9344 | +0.0061 |

Interpretation: the fine-tuned reranker did not improve this 150-question benchmark. It slightly increased lexical faithfulness but hurt source ranking and citation accuracy. Therefore, the final live demo keeps BM25 as the default retrieval configuration.

## 3. Base LLM vs Fine-Tuned LLM Smoke

Same retriever, same benchmark slice, same generation pipeline:

```text
BM25 top-3 retrieval
20 questions from eval_qa_150.jsonl
local_hf generation
```

Commands:

```bash
python scripts/evaluate_qa.py --data-dir data --retriever bm25 --generation-mode local_hf --generation-model google/flan-t5-small --top-k 3 --limit 20 --max-new-tokens 96 --output outputs/submission_eval_llm_base_flan_20.json

python scripts/evaluate_qa.py --data-dir data --retriever bm25 --generation-mode local_hf --generation-model C:\Users\bulent\Documents\Codex\2026-05-21\NLP-Term-Project-push\outputs\models\flan_t5_legal_sft_smoke_512 --top-k 3 --limit 20 --max-new-tokens 96 --output outputs/submission_eval_llm_sft_flan_20.json
```

| Metric | Base FLAN-T5-small | Fine-tuned FLAN-T5-small | Delta |
|---|---:|---:|---:|
| Token F1 | 0.1952 | 0.2319 | +0.0366 |
| ROUGE-L | 0.1871 | 0.2099 | +0.0228 |
| Citation accuracy | 0.2000 | 0.3500 | +0.1500 |
| Faithfulness proxy | 0.4861 | 0.5926 | +0.1064 |
| Top-1 source hit | 1.0000 | 1.0000 | +0.0000 |
| Top-5 source hit | 1.0000 | 1.0000 | +0.0000 |

Interpretation: the fine-tuned FLAN-T5 smoke model improves over the base FLAN-T5-small under the same pipeline. However, citation accuracy and answer quality are still weaker than the extractive answer mode, so the LLM path is kept as an optional demonstration rather than the default legal QA system.

## 4. Final System Choice

The final demo uses:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

Reason: it is the most reliable measured configuration for legal QA. Fine-tuned components were implemented and evaluated, but the final system selection is based on benchmark performance and citation reliability rather than model complexity.
