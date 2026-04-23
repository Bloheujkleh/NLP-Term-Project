# Turkish Legal QA Baseline RAG (Local)

This project is a beginner-friendly baseline Retrieval-Augmented Generation (RAG) system for Turkish legal question answering.

It supports:
- BM25 retrieval (`rank-bm25`)
- Dense retrieval (`sentence-transformers` + `FAISS`)
- Hybrid retrieval (Reciprocal Rank Fusion / RRF)
- Local answer generation (`transformers`)
- Evaluation with retrieval + answer metrics

## Project Structure

```text
NLP/
├─ data/
│  ├─ corpus.jsonl            # Dummy corpus
│  ├─ real_corpus.jsonl       # Built real corpus (generated)
│  └─ kaggle_export/          # Put local Kaggle exported files here (optional)
├─ src/
│  ├─ data_loader.py          # Build unified corpus from Kaggle + HuggingFace
│  ├─ ingest.py               # Load + chunk corpus
│  ├─ retriever.py            # BM25, Dense, Hybrid(RRF)
│  ├─ generator.py            # Local answer generator
│  ├─ metrics.py              # Recall@5, Recall@10, MRR, EM, token F1
│  └─ demo.py                 # End-to-end demo
└─ requirements.txt
```

## Data Sources

The pipeline is designed for these instructor datasets:
1. Kaggle: `batuhankalem/turkish-law-dataset-for-llm-finetuning`
2. HuggingFace: `Renicames/turkish-law-chatbot`

### Kaggle Usage (Local Export)

1. Download/export Kaggle dataset files locally.
2. Place them under:
   - `data/kaggle_export/`
3. Supported input formats:
   - `.json`, `.jsonl`, `.csv`, `.txt`

### HuggingFace Usage

`src/data_loader.py` downloads `Renicames/turkish-law-chatbot` with the `datasets` library.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Demo

```bash
python src/demo.py
```

If you want a larger evaluation set first (150 questions):

```bash
python src/build_eval_set.py
python src/demo.py
```

## Final submission pipeline (recommended)

This reduces optimistic overlap by evaluating on HF **test** questions while indexing **train+Kaggle** only:

```bash
python src/build_corpus_index.py
python src/build_eval_set.py
python src/demo.py
```

Optional fine-tuning (GPU recommended):

```bash
python src/train_embedding_contrastive.py --out_dir models/st-legal-multilingual-v1
python src/train_reranker_cross_encoder.py --out_dir models/ce-legal-v1
python src/train_lora_flan_t5.py --out_dir models/flan-t5-small-lora-legal
```

Then run evaluation with local artifacts:

```powershell
$env:DENSE_MODEL_PATH="models/st-legal-multilingual-v1"
$env:CROSS_ENCODER_PATH="models/ce-legal-v1"
$env:GEN_MODEL_PATH="models/flan-t5-small-lora-legal"
python src/demo.py
```

`demo.py` does:
1. Build/load `data/real_corpus.jsonl`
2. Chunk corpus
3. Build BM25 and Dense retrievers
4. Retrieve top-k for a sample Turkish legal question
5. Fuse rankings with RRF (Hybrid)
6. Generate final answer from retrieved contexts
7. Print retrieval/answer/groundedness metrics
8. Save all metrics to `data/results_export.json`

## Retrieval Methods (Simple Explanation)

- **BM25**: keyword-based retrieval. Strong when question terms appear in the law text.
- **Dense Retrieval**: embedding-based semantic retrieval. Helps when wording differs.
- **Hybrid (RRF)**: combines BM25 and Dense rankings for more robust top results.

## RAG Flow

1. User question
2. Retrieve relevant legal chunks (BM25/Dense/Hybrid)
3. Send top contexts + question to local generation model
4. Generate answer constrained by retrieved context

Prompt rule in generator:
- "Answer the question using only the provided legal context. If the answer is not contained in the context, say that the information is insufficient."

## Evaluation

### Retrieval
- Recall@5
- Recall@10
- MRR
- nDCG@10

### Answer Quality (tiny manual demo set)
- Exact Match (EM)
- Token-level F1
- BLEU-1 (lightweight)
- ROUGE-L (lightweight)

### Groundedness
- Faithfulness (heuristic token coverage over retrieved context)
- CitationAccuracy (whether retrieved ids include a gold source id)

### Ablation (in `demo.py`)
- Baseline (Hybrid + extractive)
- +Reranker
- +LLM
- +Reranker + LLM

## Notes

- Fully local pipeline, no paid APIs.
- First run can take time due to model downloads.
- If real datasets are not available, code falls back to available local corpora.
