# Hugging Face / Instructor Deployment

This repository contains a lightweight deployment package under:

```text
hf_space/
```

Use this folder for the instructor-facing live demo. It is intentionally separate
from the full research repository because the full `requirements.txt` includes
training and evaluation dependencies that are not needed for live custom-data
testing.

## What the Deployed App Runs

```text
Question
-> BM25 retrieval
-> source-grounded extractive answer by default
-> citation + retrieved sources
```

The deployed app defaults to `ANSWER_MODE=extractive`. This is the safest mode
for instructor custom-data testing because it is fast on CPU, always grounded in
retrieved chunks, and exposes a citation/source id for every answer.

The fine-tuned model `felinabulent/turkish-legal-qwen2-5-0-5b-rag-sft` is
available as an optional guarded generation mode. It was trained from
`Qwen/Qwen2.5-0.5B-Instruct` with LoRA on the Turkish legal RAG source-grounded
answer dataset. To enable it, set these Space environment variables:

```text
ANSWER_MODE=guarded_causal
GENERATION_MODEL=<model-id>
```

In guarded LLM mode, the selected Qwen model receives the retrieved chunks in the prompt and is
instructed to answer only from those chunks. If the generated answer is missing a
citation, too short, or weakly supported by the retrieved context, the app
automatically falls back to the extractive source-grounded answer.

The deployed app does not require GPU or external APIs. Default extractive mode
does not download a generator model and is designed to start quickly on CPU.

## Hugging Face Space Setup

1. Create a new Hugging Face Space.
2. Select **Docker** as the Space SDK.
3. Upload/copy the contents of the `hf_space/` folder into the Space root.
4. Wait for the Space build to finish.
5. Open the Space URL.

The `hf_space/` folder contains:

```text
app.py
Dockerfile
requirements.txt
README.md
data/real_corpus.jsonl
data/eval_qa_150.jsonl
scripts/demo_app.py
```

## Instructor Custom Document Test

The lower panel of the web app supports instructor-provided files.

Supported file types:

```text
.txt, .md, .csv, .json, .jsonl, .docx, .pdf, .zip
```

For document collections, upload a `.zip` containing multiple supported files.

Flow:

```text
Upload file
-> extract text
-> split into chunks
-> build temporary BM25 index
-> answer only from uploaded document chunks
-> show uploaded-file citation
```

This satisfies the interactive custom document testing requirement. If benchmark
metrics are needed on instructor data, the instructor also needs to provide a
benchmark file with `question`, `gold_answer`, and `source_id`.

The web app also provides a **Custom Benchmark Evaluation** panel. It accepts a
custom corpus/document collection plus a `.json`/`.jsonl` benchmark file and
reports Exact Match, Token F1, Top-1/Top-5 Source Hit, and Citation Accuracy.

## Local Test

From the deployment folder:

```powershell
cd hf_space
python app.py
```

Open:

```text
http://127.0.0.1:7860/ask
```

## Verified Smoke Test

The deployment package was locally smoke-tested with:

- GET `/ask`: returns the demo page.
- POST `/ask` with a legal question: returns answer and retrieved sources.
