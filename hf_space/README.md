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
-> guarded fine-tuned Qwen answer by default, with source-grounded extractive fallback
-> citation / retrieved sources
```

The default hosted mode is `ANSWER_MODE=guarded_causal`, which uses the
fine-tuned Qwen LLM first and falls back to the extractive source-grounded answer
when the generated answer is too short, missing citation, or weakly supported.
The faster extractive engine remains available from the web UI through the
**Answer engine** selector.

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

The first screen is focused on instructor-provided data. The simplest flow is:

1. Upload a single dataset `.zip`, `.json`, or `.jsonl`.
2. Click **Run Dataset**.
3. The app tries to extract documents and benchmark questions automatically and returns metrics plus sample answers.

For a `.zip`, include document files and/or JSON/JSONL files such as `corpus.jsonl`,
`documents.json`, `benchmark.json`, or `questions.jsonl`.

Manual question flow:

1. Upload a `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.docx`, `.pdf`, or `.zip` file.
   A `.zip` file can contain a document collection with multiple supported files.
2. Write a question about the uploaded document.
3. Click **Ask Uploaded Document**.
4. The system extracts text, chunks the uploaded file, builds a temporary BM25 index, and answers using only the uploaded document chunks.

The app also includes a **Upload Corpus and Benchmark** panel. The instructor can
upload a custom corpus/document collection and a benchmark `.json`/`.jsonl` file
with fields such as `question`, `gold_answer`, and `source_id`. Common aliases
such as `query`, `answer`, `answers`, `relevant_documents`, `doc_id`, and
`page_content` are also accepted. The app reports Exact Match, Answer Contains
Gold, Token F1, Top-1/Top-5 Source Hit, and Citation Accuracy for the uploaded
benchmark.

## Programmatic API

The same Space can be tested without using the web UI.

Ask over the included corpus:

```bash
curl -X POST "$SPACE_URL/api/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Kasten oldurme sucu nedir?","answer_engine":"extractive"}'
```

Ask over one uploaded document or a `.zip` document collection:

```bash
curl -X POST "$SPACE_URL/api/upload_ask" \
  -F "file=@custom_docs.zip" \
  -F "question=Kasten oldurme sucu cezasi nedir?" \
  -F "answer_engine=extractive"
```

Run a custom benchmark:

```bash
curl -X POST "$SPACE_URL/api/eval_upload" \
  -F "corpus=@custom_docs.zip" \
  -F "benchmark=@benchmark.json"
```

Run a one-file dataset evaluation:

```bash
curl -X POST "$SPACE_URL/api/dataset_eval" \
  -F "file=@dataset.zip"
```

The API returns JSON with the answer, selected answer engine, retrieved sources,
and benchmark report.

## Included Data

- `data/real_corpus.jsonl`: local Turkish legal corpus used by the default demo.
- `data/eval_qa_150.jsonl`: local custom-style benchmark reference file.

## Notes

Fine-tuned embedding, reranker, and FLAN-T5 experiments are reported in the
technical report. The hosted demo adds a more appropriate instruction-style LLM
answering path for instructor custom-document testing while keeping citation and
extractive fallback guardrails.
