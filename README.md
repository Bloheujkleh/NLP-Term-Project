# Turkish Legal RAG - CENG493

This repository contains a step-by-step implementation of a Turkish legal question answering system with Retrieval-Augmented Generation (RAG).

## Data

Expected dataset directory:

```text
Datasets_Ceng493_legal_rag/
  corpus.jsonl
  embedding.jsonl
  gold_benchmark.json
  llm.jsonl
  rag_eval.json
  reranker.jsonl
```

## Step 1: Baseline Retrieval

Dense retrieval:

```bash
python scripts/evaluate_retrieval.py --retriever dense --top-k 10
```

BM25 retrieval:

```bash
python scripts/evaluate_retrieval.py --retriever bm25 --top-k 10
```

Hybrid retrieval:

```bash
python scripts/evaluate_retrieval.py --retriever hybrid --top-k 10
```

BM25 + cross-encoder reranking:

```bash
python scripts/evaluate_reranker.py --candidate-k 50 --top-k 10
```

Optional cross-encoder fine-tuning:

```bash
python scripts/train_cross_encoder_reranker.py --epochs 1 --batch-size 8
python scripts/evaluate_reranker.py --reranker-model outputs/models/legal_cross_encoder_reranker
```

Optional embedding fine-tuning:

```bash
python scripts/train_embedding_model.py --epochs 1 --batch-size 16
python scripts/evaluate_retrieval.py --retriever dense --embedding-model outputs/models/legal_embedding_triplet
```

## Step 2: Baseline RAG Answers

This creates source-grounded extractive baseline answers from the retrieved context.

```bash
python scripts/run_baseline_rag.py --retriever hybrid --top-k 5 --limit 20
```

Richer QA evaluation with citation and faithfulness proxy metrics:

```bash
python scripts/evaluate_qa.py --retriever bm25 --generation-mode extractive --top-k 5
python scripts/analyze_qa_errors.py --input outputs/qa_eval.json
```

Outputs are written under `outputs/`.
