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
-> guarded local LLM answer generation
-> extractive fallback when the LLM answer is unsupported
-> citation / retrieved sources
```

The default hosted mode uses `Qwen/Qwen2.5-0.5B-Instruct` as a lightweight
instruction model. It receives only the retrieved legal chunks as context and is
prompted not to use outside knowledge. If the generated answer is too short,
missing a citation, or weakly supported by the retrieved chunks, the app falls
back to the extractive source-grounded answer.

The app does not require paid APIs. It can run on CPU, although first startup can
take longer while the model is downloaded.

## Instructor Custom Document Test

The lower panel of the demo supports instructor-provided documents:

1. Upload a `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.docx`, or `.pdf` file.
2. Write a question about the uploaded document.
3. Click **Ask Uploaded Document**.
4. The system extracts text, chunks the uploaded file, builds a temporary BM25 index, and answers using only the uploaded document chunks.

This is intended for interactive custom-data testing. Benchmark-style metric
evaluation additionally requires a benchmark file with `question`, `gold_answer`,
and `source_id`.

## Included Data

- `data/real_corpus.jsonl`: local Turkish legal corpus used by the default demo.
- `data/eval_qa_150.jsonl`: local custom-style benchmark reference file.

## Notes

Fine-tuned embedding, reranker, and FLAN-T5 experiments are reported in the
technical report. The hosted demo adds a more appropriate instruction-style LLM
answering path for instructor custom-document testing while keeping citation and
extractive fallback guardrails.
