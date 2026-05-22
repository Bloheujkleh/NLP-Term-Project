# 15-Minute Presentation Outline

## Slide 1 - Title

Improving Turkish Legal Question Answering with an Optimized RAG Pipeline

## Slide 2 - Motivation

- Legal QA must be grounded.
- Fluent but unsupported answers are risky.
- Evaluation should include retrieval, answer quality, citation accuracy, and faithfulness.

## Slide 3 - Dataset

- 7,579 corpus chunks
- 1,000 retrieval eval queries
- 240 gold QA questions
- 2,059 embedding triples
- 6,752 reranker pairs
- 13,758 LLM SFT examples

## Slide 4 - Architecture

```text
Question -> Retriever -> Top-k chunks -> Answer generator -> Cited answer
```

Baseline:

- BM25
- Dense MiniLM
- Hybrid retrieval
- Extractive citation-grounded answer

## Slide 5 - Retrieval Results

| Retriever | Recall@10 | MRR |
|---|---:|---:|
| BM25 | 0.975 | 0.863 |
| Dense MiniLM | 0.676 | 0.518 |
| Hybrid | 0.969 | 0.855 |

Main point: BM25 is very strong for legal text.

## Slide 6 - Reranker Experiment

| System | Recall@10 | MRR |
|---|---:|---:|
| BM25 first stage | 1.000 | 0.990 |
| Pretrained reranker | 0.810 | 0.550 |
| Fine-tuned reranker, 100q | 0.970 | 0.898 |

Main point: general-domain reranker has domain mismatch, and legal fine-tuning helps substantially.

## Slide 7 - Embedding Tuning

- `embedding.jsonl` gives query, positive passage, hard negative passage.
- Full CPU triplet-loss training was run on all 2,059 triples.
- The tuned dense model scored Recall@10 0.591 vs base dense 0.676.
- Main point: fine-tuning must be validated; it does not automatically improve retrieval.

## Slide 8 - QA Evaluation

| System | F1 | Top-5 Hit | Citation Acc. | Faithfulness |
|---|---:|---:|---:|---:|
| BM25 + extractive | 0.799 | 0.908 | 0.813 | 0.961 |

Main point: answers are grounded, but citation accuracy depends on top-1 ranking.

## Slide 9 - Error Analysis

- 22 retrieval failures
- 23 ranking failures

Implication:

- Retrieval failures need better first-stage retrieval.
- Ranking failures need domain-tuned reranker.

## Slide 10 - Optimization Plan

- Full embedding fine-tuning completed on CPU.
- Cross-encoder reranker fine-tuning completed on CPU.
- LLM instruction-tuning smoke run completed with `llm.jsonl`.
- LLM/NLI judge faithfulness completed on 240 QA examples.

## Slide 11 - Reproducibility

- GitHub repository
- Dataset split documentation
- Scripts for retrieval, QA, reranker, embedding training
- Output JSON files for metrics

## Slide 12 - Conclusion

- Built a complete Turkish legal RAG evaluation framework.
- BM25 is the strongest current demo retriever.
- Fine-tuning and judge experiments were run and measured.
- Error analysis identifies where future optimization should focus.
