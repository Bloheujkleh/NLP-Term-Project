# Step 4 - QA Generation and Evaluation

## Goal

This step evaluates the answer generation part of the RAG pipeline on `gold_benchmark.json`.

Current implemented mode:

```text
Question -> BM25 top-5 retrieval -> Extract top-1 passage -> Citation-grounded answer
```

The script also supports an optional local HuggingFace seq2seq model through `--generation-mode local_hf`.

## Evaluation Metrics

Implemented metrics:

- Exact Match
- Token F1
- ROUGE-L
- Top-1 source hit
- Top-5 source hit
- Citation label accuracy
- Lexical faithfulness proxy

The lexical faithfulness proxy measures how much of the answer body is covered by the retrieved context tokens. It is not a substitute for an LLM judge, but it is useful as a reproducible lightweight grounding signal.

## Full Baseline QA Result

Evaluation command:

```bash
python scripts/evaluate_qa.py --retriever bm25 --generation-mode extractive --top-k 5 --output outputs/qa_eval_extractive_bm25_full.json
```

Results on 240 gold benchmark questions:

| System | EM | Token F1 | ROUGE-L | Top-1 Source Hit | Top-5 Source Hit | Citation Accuracy | Faithfulness Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 + extractive answer | 0.363 | 0.799 | 0.793 | 0.813 | 0.908 | 0.813 | 0.961 |

## Error Analysis

Generated with:

```bash
python scripts/analyze_qa_errors.py --input outputs/qa_eval_extractive_bm25_full.json --output outputs/qa_error_analysis.md
```

Findings:

- Total questions: 240
- Top-5 retrieval failures: 22
- Ranking failures: 23

Interpretation:

- In 22 cases, the gold source is not retrieved in top-5. These require better first-stage retrieval, such as embedding tuning or hybrid search tuning.
- In 23 cases, the gold source is retrieved in top-5 but not ranked first. These are the best candidates for a domain-tuned reranker.
- Since the extractive baseline cites the top-1 passage, citation accuracy is bounded by top-1 source hit.

## Optional Local LLM Generation

Example command:

```bash
python scripts/evaluate_qa.py --retriever bm25 --generation-mode local_hf --generation-model google/flan-t5-small --limit 20
```

This mode is mainly for experimentation on CPU. For final results, use a stronger Turkish-capable instruction model or a fine-tuned model on GPU.

