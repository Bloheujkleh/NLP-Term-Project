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

The smoke model is not a final optimized model. It only confirms that training, saving, loading, indexing, and evaluation work end to end. A real experiment should train on all 2,059 triples and compare on the full `rag_eval.json` benchmark.

## Reporting Note

For the final report, this section can be framed as:

- Generic multilingual embeddings are not enough for Turkish legal retrieval.
- Hard-negative contrastive training is implemented and reproducible.
- Full training should be run with GPU and evaluated against the same 1,000-query benchmark.

