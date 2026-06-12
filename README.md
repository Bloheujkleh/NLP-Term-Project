# Turkish Legal RAG - CENG493

This repository contains a step-by-step implementation of a Turkish legal question answering system with Retrieval-Augmented Generation (RAG).

## Instructor Web Deployment

For a Hugging Face Space or instructor-facing hosted demo, use the lightweight
deployment package in:

```text
hf_space/
```

Create a Hugging Face Space with **Docker** SDK and upload the contents of
`hf_space/` as the Space root. This hosted version runs the reliable default
pipeline:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

It includes a `Custom Document Test` panel so the instructor can upload their
own `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.docx`, `.pdf`, or `.zip` file and ask
questions over that uploaded document or document collection. See
`HUGGINGFACE_DEPLOYMENT.md` for details.

## Instructor Quick Run

Validate the included custom-data example:

```bash
python scripts/validate_custom_data.py --data-dir sample_custom_data --require-benchmark
```

Run the reliable default demo:

```bash
python scripts/demo_app.py --data-dir data
```

The browser demo also includes a `Custom Document Test` panel. An instructor can upload a `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.docx`, `.pdf`, or `.zip` file, ask a question over that uploaded file or document collection, and inspect the retrieved chunks/citations in the same page.

The hosted Hugging Face Space defaults to the fast and robust extractive mode for
custom-data testing. The fine-tuned Qwen model is available as an optional
guarded generation mode through Space environment variables:

```text
ANSWER_MODE=guarded_causal
GENERATION_MODEL=felinabulent/turkish-legal-qwen2-5-0-5b-rag-sft
```

The Space also exposes programmatic endpoints for instructor testing:

```text
POST /api/ask
POST /api/upload_ask
POST /api/eval_upload
```

These endpoints return JSON and allow the instructor to upload a custom corpus
and benchmark without using the browser UI.

Run a submission-style Base RAG vs Fine-tuned RAG comparison after training or providing a fine-tuned reranker checkpoint:

```bash
python scripts/train_cross_encoder_reranker.py --data-dir data --epochs 1 --batch-size 4 --max-length 128 --output-dir outputs/models/legal_cross_encoder_reranker_full_cpu_128

python scripts/run_base_vs_finetuned_eval.py --data-dir data --output-dir outputs/submission_eval_bm25_vs_finetuned_reranker --base-retriever bm25 --finetuned-retriever bm25 --finetuned-reranker-model outputs/models/legal_cross_encoder_reranker_full_cpu_128
```

The fine-tuned checkpoints under `outputs/models/` are generated artifacts and are not required for the default demo. If they are not present after cloning, recreate them with the training commands above or use the documented base-model runs.

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

On the local CPU-only machine, full reranker fine-tuning was run with `batch-size 4` and `max-length 128`. It improved strongly over the pretrained reranker, but direct BM25 ranking remained best on the full retrieval benchmark. See `docs/step2_reranker_results.md`.

Optional embedding fine-tuning:

```bash
python scripts/train_embedding_model.py --epochs 1 --batch-size 16
python scripts/evaluate_retrieval.py --retriever dense --embedding-model outputs/models/legal_embedding_triplet
```

On the local CPU-only machine, full triplet fine-tuning was also run with `batch-size 4` and `max-seq-length 256`. The fine-tuned dense model scored lower than the base dense model, so the demo keeps BM25 as the primary retriever. See `docs/step3_embedding_tuning.md`.

Optional seq2seq generator SFT smoke:

```bash
python scripts/train_seq2seq_generator.py --limit 512 --eval-size 64 --epochs 1 --batch-size 2 --grad-accum 8
```

The CPU smoke run verifies the `llm.jsonl` training path, but the final demo uses the extractive generator because it is more reliable for citations.

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

Optional local generative LLM demo with the CPU fine-tuned FLAN-T5 smoke checkpoint:

```bash
python scripts/demo_app.py --data-dir data --answer-mode local_hf --generation-model outputs/models/flan_t5_legal_sft_smoke_512
```

If the generated checkpoint is not present after cloning, recreate it with:

```bash
python scripts/train_seq2seq_generator.py --data-dir data --limit 512 --eval-size 64 --epochs 1 --batch-size 2 --grad-accum 8 --output-dir outputs/models/flan_t5_legal_sft_smoke_512
```

The default demo does not require this checkpoint.

CLI version of the same generative path:

```bash
python scripts/demo_cli.py --data-dir data --question "Kasten oldurme sucu nedir?" --answer-mode local_hf --generation-model outputs/models/flan_t5_legal_sft_smoke_512
```

Fast local health check before the demo:

```bash
python scripts/smoke_test.py --data-dir data
```

## Judge-Based Faithfulness

After QA evaluation, run the local multilingual judge:

```bash
python scripts/evaluate_llm_judge.py --input outputs/qa_eval_extractive_bm25_full.json --provider nli --output outputs/nli_judge_faithfulness_full.json
```

The reported run judged 206 of 240 answers as supported by their retrieved source, for a judge faithfulness score of 0.858.

## Final Deliverables

Generated submission files are under `deliverables/`:

- `Turkish_Legal_RAG_Final_Report.docx`
- `Turkish_Legal_RAG_Presentation.pptx`

Supporting report and presentation source notes are under `docs/`.

For exact commands and measured local outputs, see `docs/reproducibility_evidence.md`.

For before/after component comparisons, see `docs/controlled_ablation_summary.md`.

For a map from project claims to implementation files, see `docs/code_evidence_map.md`.

For instructor-provided custom documents and custom benchmarks, see `docs/submission_custom_evaluation_guide.md`.

For the latest Base RAG vs Fine-tuned RAG checks aligned with the submission note, see `docs/submission_evaluation_results.md`.

Validate a custom dataset:

```bash
python scripts/validate_custom_data.py --data-dir sample_custom_data --require-benchmark
```

Run base vs fine-tuned comparison on the same benchmark:

```bash
python scripts/run_base_vs_finetuned_eval.py --data-dir sample_custom_data --output-dir outputs/sample_submission_eval
```
