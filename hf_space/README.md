---
title: Turkish Legal RAG Demo
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Turkish Legal RAG Demo

This Space runs the deployment version of the Turkish Legal RAG project.

## Live Pipeline

```text
Question
-> BM25 retrieval over Turkish legal corpus
-> source-grounded extractive answer by default
-> citation / retrieved sources
```

The default hosted mode is `ANSWER_MODE=extractive` because it is fast on CPU,
auditable, and robust for instructor-provided custom documents. The fine-tuned
LLM is still available from the web UI through the **Answer engine** selector
as `Fine-tuned Qwen LLM (slower, guarded fallback)`. It can also be enabled as
the default guarded generation mode with:

```text
ANSWER_MODE=guarded_causal
GENERATION_MODEL=felinabulent/turkish-legal-qwen2-5-0-5b-rag-sft
```

That model was trained from `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on the Turkish
legal RAG source-grounded answer dataset. To override the model, set:

```text
GENERATION_MODEL=<model-id>
```

In guarded LLM mode, the selected model receives only the retrieved legal chunks as context and is
prompted not to use outside knowledge. If the generated answer is too short,
missing a citation, or weakly supported by the retrieved chunks, the app falls
back to the extractive source-grounded answer.

The app does not require paid APIs and the default mode runs quickly on CPU.

## Instructor Custom Document Test

The lower panel of the demo supports instructor-provided documents:

1. Upload a `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.docx`, `.pdf`, or `.zip` file.
   A `.zip` file can contain a document collection with multiple supported files.
2. Write a question about the uploaded document.
3. Click **Ask Uploaded Document**.
4. The system extracts text, chunks the uploaded file, builds a temporary BM25 index, and answers using only the uploaded document chunks.

The app also includes a **Custom Benchmark Evaluation** panel. The instructor can
upload a custom corpus/document collection and a benchmark `.json`/`.jsonl` file
with fields such as `question`, `gold_answer`, and `source_id`. Common aliases
such as `query`, `answer`, `answers`, `relevant_documents`, `doc_id`, and
`page_content` are also accepted. The app reports Exact Match, Answer Contains
Gold, Token F1, Top-1/Top-5 Source Hit, and Citation Accuracy for the uploaded
benchmark.

## Included Data

- `data/real_corpus.jsonl`: local Turkish legal corpus used by the default demo.
- `data/eval_qa_150.jsonl`: local custom-style benchmark reference file.

## Notes

Fine-tuned embedding, reranker, and FLAN-T5 experiments are reported in the
technical report. The hosted demo adds a more appropriate instruction-style LLM
answering path for instructor custom-document testing while keeping citation and
extractive fallback guardrails.
