# Step 2 - Reranker Results

## Goal

The goal of this step is to test whether a cross-encoder reranker can improve the first-stage BM25 retrieval results.

Pipeline:

```text
Question -> BM25 top-k candidates -> Cross-encoder reranker -> top-10 documents
```

## Pretrained Cross-Encoder Experiment

Model:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Evaluation sample:

- 100 queries from `rag_eval.json`
- First-stage candidates: BM25 top-30
- Reranked output: top-10

| System | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 first stage on same 100 queries | 1.000 | 1.000 | 0.990 | 0.993 |
| Pretrained cross-encoder reranker | 0.700 | 0.810 | 0.550 | 0.612 |

## Interpretation

The pretrained multilingual MS MARCO reranker hurts performance on this Turkish legal dataset. This suggests a domain mismatch: the model was trained for general passage ranking, while the project data contains Turkish legal language, article numbers, court decisions, and source-specific phrasing.

This result supports the project requirement for reranker optimization. The next experiment should fine-tune the cross-encoder using `reranker.jsonl`.

## Added Scripts

- `scripts/evaluate_reranker.py`
- `scripts/train_cross_encoder_reranker.py`

## Reproducible Commands

Pretrained reranker evaluation:

```bash
python scripts/evaluate_reranker.py --limit 100 --candidate-k 30 --top-k 10 --batch-size 32 --output outputs/reranker_eval_pretrained_100.json
```

Fine-tuning command, recommended on GPU:

```bash
python scripts/train_cross_encoder_reranker.py --epochs 1 --batch-size 8
python scripts/evaluate_reranker.py --reranker-model outputs/models/legal_cross_encoder_reranker --candidate-k 50 --top-k 10
```

