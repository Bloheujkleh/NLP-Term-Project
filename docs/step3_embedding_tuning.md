# Step 3 - Embedding Tuning

## Goal

The generic multilingual dense retriever underperformed BM25 on the full retrieval benchmark:

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense, multilingual MiniLM | 0.613 | 0.676 | 0.518 | 0.556 |
| BM25 | 0.947 | 0.975 | 0.863 | 0.890 |

This motivates domain adaptation for the embedding model.

## Training Data

`embedding.jsonl` contains 2,059 training triples:

```text
query
positive_passage
negative_passage
```

Most negatives are hard negatives from the same source or legal category, which is useful for contrastive training.

## Implemented Training Script

The script `scripts/train_embedding_model.py` fine-tunes a `SentenceTransformer` bi-encoder with triplet loss.

Recommended command, preferably on GPU:

```bash
python scripts/train_embedding_model.py --epochs 1 --batch-size 16 --max-seq-length 384
```

Evaluation after training:

```bash
python scripts/evaluate_retrieval.py --retriever dense --embedding-model outputs/models/legal_embedding_triplet --top-k 10
```

## CPU Smoke Test

A tiny smoke training run was executed locally to verify the training and evaluation pipeline:

```bash
python scripts/train_embedding_model.py --limit 64 --epochs 1 --batch-size 8 --max-seq-length 256 --output-dir outputs/models/legal_embedding_triplet_smoke
python scripts/evaluate_retrieval.py --retriever dense --embedding-model outputs/models/legal_embedding_triplet_smoke --limit 50 --top-k 10
```

Results on the first 50 retrieval queries:

| Model | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Base dense model | 0.820 | 0.900 | 0.738 | 0.776 |
| Smoke fine-tuned model, 64 triples | 0.840 | 0.880 | 0.725 | 0.762 |

The smoke model is not a final optimized model. It only confirms that training, saving, loading, indexing, and evaluation work end to end.

## Full CPU Fine-Tuning Run

Full embedding fine-tuning was also executed locally on CPU-only PyTorch. The environment did not expose a CUDA device, so the run used a smaller batch size and sequence length than the preferred GPU configuration.

Training command:

```bash
python scripts/train_embedding_model.py --epochs 1 --batch-size 4 --max-seq-length 256 --output-dir outputs/models/legal_embedding_triplet_full_cpu
```

Training summary:

| Setting | Value |
|---|---:|
| Training triples | 2,059 |
| Epochs | 1 |
| Batch size | 4 |
| Max sequence length | 256 |
| Runtime | 2,471 seconds |
| Training loss | 3.257 |

Evaluation on the full 1,000-query `rag_eval.json` benchmark:

| Dense model | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Base multilingual MiniLM | 0.613 | 0.676 | 0.518 | 0.556 |
| CPU triplet fine-tuned model | 0.539 | 0.591 | 0.456 | 0.488 |

The naive triplet fine-tuning run degraded dense retrieval quality. This is an important ablation result: fine-tuning is not automatically beneficial, and the embedding objective, hard-negative sampling, sequence length, and validation strategy must be tuned carefully. For the final demo system, BM25 remains the strongest retriever.

## Reporting Note

For the final report, this section can be framed as:

- Generic multilingual embeddings are not enough for Turkish legal retrieval.
- Hard-negative contrastive training is implemented and reproducible.
- A full CPU fine-tuning run was completed, but it did not improve dense retrieval.
- Future work should tune the embedding loss, hard-negative sampling, context length, and base Turkish legal embedding model.
