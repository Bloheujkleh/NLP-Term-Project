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
-> extractive source-grounded answer
-> citation / retrieved sources
```

The deployment intentionally uses the lightweight BM25 + extractive configuration.
It does not require GPU, paid APIs, or downloading large language models.

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
technical report. The live deployment uses the most reliable measured demo
configuration for source-grounded legal QA.
