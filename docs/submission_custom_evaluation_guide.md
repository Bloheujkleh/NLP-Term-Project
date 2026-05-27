# Submission Custom Evaluation Guide

This guide explains how the instructor can run the RAG system on a custom document collection and a custom benchmark.

## 1. Custom Document Collection

Create a folder with one of these corpus files:

```text
custom_data/
  corpus.jsonl
```

The system also accepts `real_corpus.jsonl` or `corpus_index.jsonl`. Each line must be one JSON object:

```json
{"id":"DOC_001","title":"Turk Ceza Kanunu Madde 81","text":"Kasten oldurme sucunu isleyen kisi muebbet hapis cezasi ile cezalandirilir.","metadata":{"citation_label":"TCK Madde 81 - DOC_001"}}
```

Required fields:

| Field | Meaning |
|---|---|
| `id` | Unique document/passage id. Benchmark source ids must match this value. |
| `text` | Legal passage text used for retrieval and answer grounding. |

Optional fields:

| Field | Meaning |
|---|---|
| `title` | Human-readable source title. |
| `metadata.citation_label` | Citation shown in the final answer. |

## 2. Custom Benchmark

Add one benchmark file to the same folder:

```text
custom_data/
  eval_qa.jsonl
```

The system also accepts `custom_benchmark.jsonl`, `benchmark.jsonl`, or `eval_qa_150.jsonl`.

Recommended format:

```json
{"question":"Kasten oldurme sucu nedir?","gold_answer":"Kasten oldurme sucunu isleyen kisi muebbet hapis cezasi ile cezalandirilir.","source_id":"DOC_001"}
```

Accepted aliases:

| Concept | Accepted fields |
|---|---|
| Question | `question` or `query` |
| Gold answer | `gold_answer`, `verified_answer`, or `answer` |
| Relevant documents | `source_id`, `gold_source_id`, `gold_chunk_ids`, `relevant_documents`, `relevant_doc_ids`, or `gold_sources[].corpus_row_id` |

## 3. Validate Custom Data

Before running the system, validate the custom folder:

```bash
python scripts/validate_custom_data.py --data-dir custom_data --require-benchmark
```

The included sample can be checked with:

```bash
python scripts/validate_custom_data.py --data-dir sample_custom_data --require-benchmark
```

## 4. Run The Demo On Custom Documents

Browser demo:

```bash
python scripts/demo_app.py --data-dir custom_data
```

CLI demo:

```bash
python scripts/demo_cli.py --data-dir custom_data --question "Kasten oldurme sucu nedir?"
```

The default demo uses:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

This is the recommended live legal demo because it is the most citation-reliable mode.

## 5. Base RAG vs Fine-Tuned RAG Evaluation

Run both systems on the same corpus, benchmark, and answer generator:

```bash
python scripts/run_base_vs_finetuned_eval.py --data-dir custom_data --output-dir outputs/custom_submission_eval
```

This default command validates the evaluation wrapper but does not include a fine-tuned component unless a checkpoint is provided. For a real fine-tuned RAG comparison, provide the reranker checkpoint:

```bash
python scripts/run_base_vs_finetuned_eval.py \
  --data-dir custom_data \
  --output-dir outputs/custom_submission_eval \
  --base-retriever bm25 \
  --finetuned-retriever bm25 \
  --finetuned-reranker-model outputs/models/legal_cross_encoder_reranker_full_cpu_128
```

For a fine-tuned embedding comparison, provide a fine-tuned embedding checkpoint and choose `dense` or `hybrid`:

```bash
python scripts/run_base_vs_finetuned_eval.py \
  --data-dir custom_data \
  --output-dir outputs/custom_embedding_eval \
  --base-retriever dense \
  --base-embedding-model intfloat/multilingual-e5-base \
  --finetuned-retriever dense \
  --finetuned-embedding-model outputs/models/legal_embedding_triplet_full_cpu
```

For the optional same-LLM generator comparison:

```bash
python scripts/run_base_vs_finetuned_eval.py \
  --data-dir custom_data \
  --output-dir outputs/custom_llm_eval \
  --generation-mode local_hf \
  --generation-model outputs/models/flan_t5_legal_sft_smoke_512
```

The script writes:

```text
outputs/custom_submission_eval/base_rag_qa.json
outputs/custom_submission_eval/finetuned_rag_qa.json
outputs/custom_submission_eval/base_vs_finetuned_summary.json
```

## 6. Metrics

When a gold benchmark is available, the project reports:

| Metric | Why it is used |
|---|---|
| `top1_source_hit` / `top5_source_hit` | Checks whether the relevant document was retrieved. |
| `citation_label_accuracy` | Checks whether the answer cites the expected source. |
| `token_f1` / `rouge_l` | Checks lexical overlap with the gold answer. |
| `faithfulness_proxy` | Checks whether answer tokens are supported by retrieved contexts. |

Retrieval-only evaluation can also be run:

```bash
python scripts/evaluate_retrieval.py --data-dir custom_data --retriever bm25 --top-k 10
```

## 7. Important Framing For Evaluation

The default final demo is not the most complex pipeline; it is the most reliable measured pipeline:

```text
BM25 retrieval + extractive source-grounded answer + citation
```

Fine-tuned embedding, reranker, and FLAN-T5 generator experiments are included for ablation and comparison. If a fine-tuned component does not improve the benchmark score, that result is reported rather than hidden.
