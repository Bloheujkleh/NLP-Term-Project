# Reproducibility Evidence

This document records the exact local commands and measured outputs used in the report. Large generated artifacts under `outputs/` and model checkpoints are intentionally not committed to GitHub, but the scripts are reproducible.

## Environment Check

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Observed:

```text
torch 2.12.0+cpu
cuda_available False
cuda_count 0
```

## Retrieval Evaluation

BM25:

```bash
python scripts/evaluate_retrieval.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --retriever bm25 --top-k 10 --output outputs/retrieval_eval_bm25_full.json
```

Result:

```text
Recall@5 0.947
Recall@10 0.975
MRR 0.863
nDCG@10 0.890
```

Dense:

```bash
python scripts/evaluate_retrieval.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --retriever dense --top-k 10 --output outputs/retrieval_eval_dense_full.json
```

Result:

```text
Recall@5 0.613
Recall@10 0.676
MRR 0.518
nDCG@10 0.556
```

Hybrid:

```bash
python scripts/evaluate_retrieval.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --retriever hybrid --dense-weight 0.35 --top-k 10 --output outputs/retrieval_eval_hybrid_full.json
```

Result:

```text
Recall@5 0.940
Recall@10 0.969
MRR 0.855
nDCG@10 0.883
```

## QA Evaluation

```bash
python scripts/evaluate_qa.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --retriever bm25 --generation-mode extractive --top-k 5 --output outputs/qa_eval_extractive_bm25_full.json
```

Result:

```text
Exact Match 0.363
Token F1 0.799
ROUGE-L 0.793
Top-1 Source Hit 0.813
Top-5 Source Hit 0.908
Citation Accuracy 0.813
Faithfulness Proxy 0.961
```

## Embedding Fine-Tuning

```bash
python scripts/train_embedding_model.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --epochs 1 --batch-size 4 --max-seq-length 256 --output-dir outputs\models\legal_embedding_triplet_full_cpu
```

Observed:

```text
train_runtime 2471 sec
train_loss 3.257
```

Evaluation:

```bash
python scripts/evaluate_retrieval.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --retriever dense --embedding-model outputs\models\legal_embedding_triplet_full_cpu --top-k 10 --batch-size 128 --output outputs\retrieval_eval_dense_finetuned_full_cpu.json
```

Result:

```text
Recall@5 0.539
Recall@10 0.591
MRR 0.456
nDCG@10 0.488
```

## Reranker Fine-Tuning

```bash
python scripts/train_cross_encoder_reranker.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --epochs 1 --batch-size 4 --max-length 128 --output-dir outputs\models\legal_cross_encoder_reranker_full_cpu_128
```

Observed:

```text
train_runtime 3047 sec
train_loss 0.358
```

Full evaluation:

```bash
python scripts/evaluate_reranker.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --reranker-model outputs\models\legal_cross_encoder_reranker_full_cpu_128 --candidate-k 50 --top-k 10 --batch-size 16 --max-length 128 --output outputs\reranker_eval_finetuned_full_cpu_128_full.json
```

Result:

```text
BM25 first-stage Recall@10 0.975
Fine-tuned reranker Recall@10 0.915
Fine-tuned reranker MRR 0.789
Fine-tuned reranker nDCG@10 0.820
```

## NLI/Judge Faithfulness

```bash
python scripts/evaluate_llm_judge.py --input outputs\qa_eval_extractive_bm25_full.json --provider nli --output outputs\nli_judge_faithfulness_full.json
```

Result:

```text
Judge-supported answers 206 / 240
Judge-rejected answers 34 / 240
Judge faithfulness 0.858
```

## LLM/SFT Smoke

Base FLAN-T5-small QA check:

```bash
python scripts/evaluate_qa.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --retriever bm25 --generation-mode local_hf --generation-model google/flan-t5-small --top-k 3 --limit 20 --max-new-tokens 96 --output outputs\qa_eval_local_hf_base_flan_20.json
```

Result:

```text
Token F1 0.076
ROUGE-L 0.056
Citation Accuracy 0.000
Faithfulness Proxy 0.452
```

Fine-tuning command:

```bash
python scripts/train_seq2seq_generator.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --limit 512 --eval-size 64 --epochs 1 --batch-size 2 --grad-accum 8 --max-input-length 512 --max-target-length 160 --output-dir outputs\models\flan_t5_legal_sft_smoke_512 --metrics-output outputs\llm_sft_smoke_512_metrics.json
```

Observed:

```text
training_examples 448
eval_examples 64
train_runtime 629 sec
train_loss 7.368
eval_loss 0.514
```

Small QA check:

```bash
python scripts/evaluate_qa.py --data-dir C:\Users\bulent\Documents\Codex\2026-05-21\nlp-or\Datasets_Ceng493_legal_rag --retriever bm25 --generation-mode local_hf --generation-model outputs\models\flan_t5_legal_sft_smoke_512 --top-k 3 --limit 20 --max-new-tokens 96 --output outputs\qa_eval_local_hf_sft_smoke_20.json
```

Result:

```text
Token F1 0.103
ROUGE-L 0.075
Citation Accuracy 0.000
Faithfulness Proxy 0.616
```

This is why the live demo keeps the extractive source-grounded generator instead of the small CPU-trained FLAN-T5 model.
