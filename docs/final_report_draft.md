# Improving Turkish Legal Question Answering with an Optimized RAG Pipeline

## 1. Introduction

This project develops a domain-adapted Retrieval-Augmented Generation (RAG) system for Turkish legal question answering. The goal is to answer Turkish legal questions with grounded, source-supported, and citation-consistent responses while minimizing hallucination.

Legal QA is especially sensitive to hallucination because an answer may look fluent while being unsupported by legal sources. For this reason, the system is evaluated not only by answer similarity metrics, but also by retrieval quality, citation accuracy, and faithfulness-oriented grounding signals.

## 2. Problem Definition

Input:

```text
A Turkish legal question
```

Output:

```text
A Turkish answer grounded in retrieved legal sources, with citation information
```

The main research questions are:

- How strong is a simple lexical baseline for Turkish legal retrieval?
- Does a generic multilingual dense embedding model perform well in this domain?
- Does hybrid retrieval improve over BM25?
- Does a pretrained multilingual reranker improve ranking quality?
- Which errors come from retrieval failure and which come from ranking failure?

## 3. Dataset

The project uses the provided Turkish legal RAG dataset.

| File | Size | Purpose |
|---|---:|---|
| `corpus.jsonl` | 7,579 rows | Source chunks used for retrieval |
| `rag_eval.json` | 1,000 rows | Retrieval benchmark with gold chunk IDs |
| `gold_benchmark.json` | 240 rows | Gold QA benchmark with verified answers and gold sources |
| `embedding.jsonl` | 2,059 rows | Query-positive-negative triples for embedding tuning |
| `reranker.jsonl` | 6,752 rows | Query-passage-label pairs for reranker tuning |
| `llm.jsonl` | 13,758 rows | Source-grounded instruction tuning examples |

All 240 gold benchmark source IDs are present in the retrieval corpus, so the benchmark fits Scenario 1 from the rubric: Gold Question + Gold Answer + Gold Document.

## 4. System Architecture

The baseline RAG pipeline is:

```text
Question -> Retriever -> Top-k source chunks -> Answer generator -> Citation-grounded answer
```

Implemented retrievers:

- BM25 lexical retrieval
- Dense retrieval with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Hybrid retrieval using min-max normalized BM25 and dense scores

Implemented answer generation modes:

- Extractive grounded answer from the top retrieved source
- Optional local HuggingFace seq2seq generation mode

## 5. Evaluation Methodology

### 5.1 Retrieval Metrics

Retrieval is evaluated on `rag_eval.json` using:

- Recall@5
- Recall@10
- Mean Reciprocal Rank
- nDCG@10

### 5.2 QA Metrics

Question answering is evaluated on `gold_benchmark.json` using:

- Exact Match
- Token F1
- ROUGE-L
- Top-1 source hit
- Top-5 source hit
- Citation label accuracy
- Lexical faithfulness proxy

Citation label accuracy checks whether the generated answer contains the gold citation label. The lexical faithfulness proxy measures how much of the answer body is covered by retrieved context tokens.

## 6. Experiments and Results

### 6.1 Retrieval Baselines

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.947 | 0.975 | 0.863 | 0.890 |
| Dense multilingual MiniLM | 0.613 | 0.676 | 0.518 | 0.556 |
| Hybrid, dense weight 0.35 | 0.940 | 0.969 | 0.855 | 0.883 |

BM25 is the strongest retrieval baseline. This is likely because Turkish legal questions often include exact legal terms, article numbers, court names, or source-specific wording.

The generic multilingual dense model performs substantially worse than BM25, which motivates domain-specific embedding adaptation.

The first hybrid configuration does not improve over BM25. This suggests that hybrid retrieval requires careful weighting and that adding a weak dense retriever can reduce ranking quality.

### 6.2 Reranker Experiments

Pipeline:

```text
Question -> BM25 top-30 -> cross-encoder reranker -> top-10
```

Model:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Evaluation was performed on a 100-query subset due to CPU-only inference cost.

| System | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 first stage on same 100 queries | 1.000 | 1.000 | 0.990 | 0.993 |
| Pretrained cross-encoder reranker | 0.700 | 0.810 | 0.550 | 0.612 |

The pretrained reranker hurts performance. This indicates domain mismatch: the reranker was trained for general passage ranking, not Turkish legal ranking. This result supports the requirement for cross-encoder fine-tuning using `reranker.jsonl`.

A full CPU fine-tuning run was then performed on all 6,752 rows of `reranker.jsonl` using one epoch, batch size 4, and max sequence length 128. Training took 3,047 seconds and reached training loss 0.358.

Full benchmark evaluation:

| System | Queries | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 first stage | 1,000 | 0.947 | 0.975 | 0.864 | 0.890 |
| Fine-tuned reranker | 1,000 | 0.882 | 0.915 | 0.789 | 0.820 |

The fine-tuned reranker substantially improves over the pretrained reranker, but it still does not beat the raw BM25 ranking. Therefore, the final demo uses BM25 directly while reporting reranker fine-tuning as an ablation.

### 6.3 Embedding Tuning Experiment

A triplet-loss embedding fine-tuning script was implemented using `embedding.jsonl`.

Training triple:

```text
query, positive_passage, hard_negative_passage
```

First, a small CPU smoke test used only 64 triples to verify the training and evaluation workflow.

| Model | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Base dense model, first 50 queries | 0.820 | 0.900 | 0.738 | 0.776 |
| Smoke fine-tuned model, 64 triples | 0.840 | 0.880 | 0.725 | 0.762 |

The smoke run confirmed that the training pipeline works end to end.

A full CPU fine-tuning run was then executed using all 2,059 embedding triples. The local PyTorch installation did not expose a CUDA device, so the run used `batch_size=4`, `max_seq_length=256`, and one epoch. Training took 2,471 seconds and reached training loss 3.257.

Full benchmark evaluation:

| Dense model | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Base multilingual MiniLM | 0.613 | 0.676 | 0.518 | 0.556 |
| CPU triplet fine-tuned model | 0.539 | 0.591 | 0.456 | 0.488 |

The full CPU fine-tuning run degraded dense retrieval quality. This is an important ablation result: fine-tuning is not automatically beneficial, especially when the base model, negative sampling, objective, sequence length, and validation strategy are not tuned for Turkish legal retrieval. Therefore, the final demo system keeps BM25 as the primary retriever.

### 6.4 QA and Grounding Evaluation

The current QA baseline retrieves top-5 chunks with BM25 and returns an extractive source-grounded answer from the top-1 chunk.

| System | EM | Token F1 | ROUGE-L | Top-1 Source Hit | Top-5 Source Hit | Citation Accuracy | Faithfulness Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 + extractive answer | 0.363 | 0.799 | 0.793 | 0.813 | 0.908 | 0.813 | 0.961 |

The faithfulness proxy is high because the baseline answer is extractive and directly uses retrieved source text. However, answer quality and citation accuracy are limited by whether the correct source is ranked first.

### 6.5 LLM/NLI Judge Faithfulness

To add a semantic grounding check beyond token overlap, a judge-based faithfulness evaluator was implemented in `scripts/evaluate_llm_judge.py`. The script supports an optional API-based LLM judge, a local FLAN-T5 judge, and a local multilingual NLI judge. Because no API key was available and FLAN-T5-small was unreliable for Turkish legal text, the final run used:

```text
MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli
```

The judge compares each generated answer against the top retrieved source text and predicts whether the answer is entailed by that source.

| Judge | Examples | Supported | Rejected | Faithfulness |
|---|---:|---:|---:|---:|
| Multilingual NLI judge | 240 | 206 | 34 | 0.858 |

The judge score is lower than the lexical proxy, which is expected because semantic entailment is stricter than token overlap. It still indicates that most generated answers are grounded in the retrieved source.

## 7. Error Analysis

Error analysis on the 240-question gold benchmark found:

- Top-5 retrieval failures: 22
- Ranking failures: 23

Failure definitions:

- Retrieval failure: the gold source is not present in top-5.
- Ranking failure: the gold source is in top-5 but not top-1.
- Formatting mismatch: the correct source is retrieved but answer wording differs from the verified answer.

These results show that future improvements should target two different parts of the pipeline:

- Better first-stage retrieval for the 22 retrieval failures
- Domain-tuned reranking for the 23 ranking failures

## 8. Reproducibility

Main commands:

```bash
python scripts/evaluate_retrieval.py --retriever bm25 --top-k 10 --output outputs/retrieval_eval_bm25_full.json
python scripts/evaluate_retrieval.py --retriever dense --top-k 10 --output outputs/retrieval_eval_dense_full.json
python scripts/evaluate_retrieval.py --retriever hybrid --dense-weight 0.35 --top-k 10 --output outputs/retrieval_eval_hybrid_full.json
python scripts/evaluate_qa.py --retriever bm25 --generation-mode extractive --top-k 5 --output outputs/qa_eval_extractive_bm25_full.json
python scripts/evaluate_llm_judge.py --input outputs/qa_eval_extractive_bm25_full.json --provider nli --output outputs/nli_judge_faithfulness_full.json
python scripts/evaluate_reranker.py --reranker-model outputs/models/legal_cross_encoder_reranker_full_cpu_128 --candidate-k 50 --top-k 10 --batch-size 16 --max-length 128 --output outputs/reranker_eval_finetuned_full_cpu_128_full.json
python scripts/analyze_qa_errors.py --input outputs/qa_eval_extractive_bm25_full.json --output outputs/qa_error_analysis.md
```

Training commands:

```bash
python scripts/train_embedding_model.py --epochs 1 --batch-size 16 --max-seq-length 384
python scripts/train_cross_encoder_reranker.py --epochs 1 --batch-size 8
```

## 9. Hardware and Limitations

The local environment used for these experiments is CPU-only. CUDA is not available. Because of this, full embedding and cross-encoder reranker fine-tuning were run with smaller CPU configurations, while large LLM fine-tuning was prepared as reproducible data and scripts but not fully executed locally.

Limitations:

- The current answer generator is extractive, not a final fine-tuned LLM.
- CPU embedding fine-tuning degraded dense retrieval, so the final demo uses BM25.
- CPU reranker fine-tuning improved over the pretrained reranker but still did not beat BM25.
- Full LLM fine-tuning should be run on GPU.

## 10. Conclusion

The project establishes a reproducible Turkish legal RAG pipeline with retrieval, QA evaluation, reranker testing, embedding fine-tuning infrastructure, and error analysis.

The strongest current system is BM25-based retrieval with extractive grounded answers. BM25 achieves 0.975 Recall@10 on the retrieval benchmark and 0.908 Top-5 source hit on the gold QA benchmark.

The experiments also reveal clear optimization directions. Generic dense embeddings underperform lexical retrieval, a naive CPU triplet fine-tuning run further degrades dense retrieval, and a pretrained general-domain reranker hurts performance. Fine-tuning the reranker improves it substantially, but the direct BM25 ranking remains strongest on this benchmark. Therefore, future optimized systems should use carefully validated domain adaptation rather than assuming that fine-tuning alone will improve the pipeline.
