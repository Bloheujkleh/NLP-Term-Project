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

This result supports the project requirement for reranker optimization.

## Full CPU Fine-Tuned Reranker

The cross-encoder reranker was fine-tuned locally on CPU using all 6,752 rows from `reranker.jsonl`.

Training configuration:

| Setting | Value |
|---|---:|
| Training pairs | 6,752 |
| Epochs | 1 |
| Batch size | 4 |
| Max sequence length | 128 |
| Runtime | 3,047 seconds |
| Training loss | 0.358 |

Full benchmark evaluation used BM25 top-50 as first-stage candidates and reranked the final top-10.

| System | Queries | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 first stage | 1,000 | 0.947 | 0.975 | 0.864 | 0.890 |
| Fine-tuned reranker | 1,000 | 0.882 | 0.915 | 0.789 | 0.820 |

The fine-tuned reranker substantially improves over the pretrained reranker, especially on the 100-query comparison where Recall@10 increases from 0.810 to 0.970. However, it still does not beat the direct BM25 ranking on the full benchmark. This means domain fine-tuning helps, but the final demo keeps BM25 as the safest primary ranking method.

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

CPU fine-tuning command used in this run:

```bash
python scripts/train_cross_encoder_reranker.py --epochs 1 --batch-size 4 --max-length 128 --output-dir outputs/models/legal_cross_encoder_reranker_full_cpu_128
python scripts/evaluate_reranker.py --reranker-model outputs/models/legal_cross_encoder_reranker_full_cpu_128 --candidate-k 50 --top-k 10 --batch-size 16 --max-length 128 --output outputs/reranker_eval_finetuned_full_cpu_128_full.json
```
