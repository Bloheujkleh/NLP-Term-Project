# Step 1 - Baseline RAG Results

## Dataset

- Corpus: 7,579 Turkish legal chunks
- Retrieval evaluation set: 1,000 queries with gold chunk ids
- Gold QA benchmark: 240 questions with verified answers and gold source documents

## Retrieval Results

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.947 | 0.975 | 0.863 | 0.890 |
| Dense, multilingual MiniLM | 0.613 | 0.676 | 0.518 | 0.556 |
| Hybrid, dense weight 0.35 | 0.940 | 0.969 | 0.855 | 0.883 |

## Baseline QA Results

Baseline answer generation is extractive: the system returns the top retrieved source passage with its citation.

| Retriever | Questions | EM | Token F1 | Top-1 Source Hit |
|---|---:|---:|---:|---:|
| BM25 | 240 | 0.363 | 0.799 | 0.813 |

## Initial Observations

- BM25 is a very strong baseline for this dataset, likely because many questions contain exact legal terms, article numbers, court names, or source-specific wording.
- The generic multilingual dense embedding model underperforms BM25. This motivates domain adaptation or contrastive fine-tuning for the embedding layer.
- The first hybrid configuration does not improve over BM25. Hybrid retrieval should be tuned with different dense weights and candidate sizes.
- The baseline answer generator is grounded and citation-aware, but it is not yet a true generative legal assistant. The next step is to add a reranker and then an LLM-based answer generator.

## Output Files

- `outputs/retrieval_eval_bm25_full.json`
- `outputs/retrieval_eval_dense_full.json`
- `outputs/retrieval_eval_hybrid_full.json`
- `outputs/baseline_rag_bm25_full.json`

