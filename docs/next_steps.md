# Next Steps

## Step 2 - Reranker

Train or load a cross-encoder reranker using `reranker.jsonl`, then evaluate:

- BM25 top-50 candidates
- Cross-encoder rerank to top-10
- Compare Recall@5, Recall@10, MRR, nDCG@10 against Step 1

Status: pretrained reranker evaluation is implemented. Full fine-tuning is recommended on GPU because no CUDA device is available in the current environment.

## Step 3 - Embedding Tuning

Use `embedding.jsonl` for contrastive fine-tuning:

- Query
- Positive passage
- Hard negative passage

Then rebuild the FAISS index and compare dense retrieval before and after tuning.

Status: training and evaluation scripts are implemented. A CPU smoke run succeeded, but full training is recommended on GPU.

## Step 4 - LLM Answer Generation

Use `llm.jsonl` for instruction tuning or as prompt examples:

- Source-grounded answers
- Short summaries with citation
- Source-limited explanations

The evaluation should include EM/F1 plus citation accuracy and hallucination analysis.

Status: QA generation/evaluation script is implemented with extractive and optional local HuggingFace modes. Error analysis is generated from the full 240-question gold benchmark.

## Step 5 - Final Optimized Pipeline

Recommended final comparison table:

| System | Recall@10 | MRR | QA F1 | Citation Accuracy | Faithfulness |
|---|---:|---:|---:|---:|---:|
| Baseline RAG | | | | | |
| + Embedding tuning | | | | | |
| + Reranker | | | | | |
| + LLM fine-tuning | | | | | |
| Fully optimized | | | | | |
