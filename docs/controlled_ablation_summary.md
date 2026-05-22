# Controlled Ablation Summary

This file clarifies the scientific comparison setup: each ablation keeps the pipeline fixed and changes only one component at a time where possible.

## A. Dense Retrieval: Base vs Fine-Tuned Embedding

Pipeline:

```text
Question -> Dense embedding -> FAISS vector search -> top-10 chunks
```

Only changed component:

```text
base embedding model -> triplet fine-tuned embedding model
```

| Dense model | Eval queries | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Base multilingual MiniLM | 1,000 | 0.613 | 0.676 | 0.518 | 0.556 |
| CPU triplet fine-tuned MiniLM | 1,000 | 0.539 | 0.591 | 0.456 | 0.488 |

Interpretation: the naive triplet fine-tuning setup did not improve dense retrieval. This is why the final demo does not use the fine-tuned dense retriever.

## B. Reranker: Pretrained vs Fine-Tuned Cross-Encoder

Pipeline:

```text
Question -> BM25 top-50 candidates -> cross-encoder reranker -> top-10 chunks
```

Only changed component:

```text
pretrained cross-encoder -> legal fine-tuned cross-encoder
```

100-query comparison:

| Reranker | Eval queries | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Pretrained MS MARCO multilingual reranker | 100 | 0.700 | 0.810 | 0.550 | 0.612 |
| CPU fine-tuned legal reranker | 100 | 0.950 | 0.970 | 0.898 | 0.916 |

Full benchmark:

| System | Eval queries | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 first-stage ranking | 1,000 | 0.947 | 0.975 | 0.864 | 0.890 |
| CPU fine-tuned legal reranker | 1,000 | 0.882 | 0.915 | 0.789 | 0.820 |

Interpretation: fine-tuning substantially improves the reranker compared with the pretrained version. However, direct BM25 ranking remains strongest on the full benchmark.

## C. LLM Generator: Base vs SFT Smoke

Pipeline:

```text
Question -> BM25 top-3 chunks -> FLAN-T5 generator -> answer
```

Only changed component:

```text
base FLAN-T5-small -> FLAN-T5-small SFT smoke model
```

Both models were evaluated on the same first 20 gold QA examples.

| Generator | Eval questions | Token F1 | ROUGE-L | Citation Accuracy | Faithfulness Proxy |
|---|---:|---:|---:|---:|---:|
| Base FLAN-T5-small | 20 | 0.076 | 0.056 | 0.000 | 0.452 |
| SFT smoke FLAN-T5-small | 20 | 0.103 | 0.075 | 0.000 | 0.616 |

Interpretation: the SFT smoke model improves over the base FLAN-T5-small under the same generation pipeline, but both are too weak for citation-reliable legal QA. Therefore, the final demo uses the extractive grounded generator.

## D. Final Demo Selection

The final live demo is not claimed to be the fully optimized generative LLM system. It is the most reliable measured configuration for citation-grounded legal QA:

```text
BM25 retrieval -> extractive grounded answer -> citation
```

This choice is based on the controlled ablations above: dense fine-tuning did not help, reranker fine-tuning improved but did not beat BM25, and the small SFT generator was not citation-reliable enough for live use.
