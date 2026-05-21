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

On the local CPU-only machine, full triplet fine-tuning was also run with `batch-size 4` and `max-seq-length 256`. The fine-tuned dense model scored lower than the base dense model, so the demo keeps BM25 as the primary retriever. See `docs/step3_embedding_tuning.md`.

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

## Live Demo

For the instructor demo, use the dependency-light browser demo:

```bash
python scripts/demo_app.py --data-dir data
```

On Windows, the same demo can be launched with:

```bash
run_demo.bat
```

Then open:

```text
http://127.0.0.1:7860
```

Reliable demo questions for the included small corpus:

```text
Kasten oldurme sucu nedir?
Adil yargilanma hakki nasil guvence altina alinir?
Evlilik birligi temelinden sarsilirsa ne olur?
```

The demo shows:

- the generated grounded answer,
- the retrieved source passages,
- the citation/source label,
- and the headline evaluation metrics.

Terminal-only fallback:

```bash
python scripts/demo_cli.py --data-dir data --question "Kasten oldurme sucu nedir?"
```

## Final Deliverables

Generated submission files are under `deliverables/`:

- `Turkish_Legal_RAG_Final_Report.docx`
- `Turkish_Legal_RAG_Presentation.pptx`

Supporting report and presentation source notes are under `docs/`.
