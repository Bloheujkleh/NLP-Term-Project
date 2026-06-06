# Live Demo Runbook

## Goal

The live demo should show that the system is a working RAG pipeline, not only a report:

```text
question -> retrieval -> source-grounded answer -> citation
```

## Start Demo

```bash
python scripts/demo_app.py --data-dir data
```

Open:

```text
http://127.0.0.1:7860
```

## Custom Document Upload

The same browser page has a `Custom Document Test` panel. Use it when the instructor wants to upload a file during or after the demo:

1. Choose a `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.docx`, or `.pdf` file.
2. Write a question about the uploaded file.
3. Click `Ask Uploaded Document`.
4. The app builds a temporary BM25 index over the uploaded file, returns a source-grounded extractive answer, and lists the retrieved uploaded-file chunks as citations.

If the browser app fails for any reason, use the CLI fallback:

```bash
python scripts/demo_cli.py --data-dir data --question "Kasten oldurme sucu nedir?"
```

## Optional Generative LLM Demo

The default browser demo uses the reliable extractive grounded generator. To show that a real local LLM path is also connected, run:

```bash
python scripts/demo_app.py --data-dir data --answer-mode local_hf --generation-model outputs/models/flan_t5_legal_sft_smoke_512
```

CLI check:

```bash
python scripts/demo_cli.py --data-dir data --question "Kasten oldurme sucu nedir?" --answer-mode local_hf --generation-model outputs/models/flan_t5_legal_sft_smoke_512
```

Use this as an optional demonstration only. The FLAN-T5 smoke model is CPU fine-tuned and small, so the final answer quality is weaker than the extractive mode.

## Recommended Demo Questions

Use these with the included small corpus because they retrieve clean sources:

```text
Kasten oldurme sucu nedir?
Adil yargilanma hakki nasil guvence altina alinir?
Evlilik birligi temelinden sarsilirsa ne olur?
```

## What To Say During Demo

1. The question is sent to the retriever.
2. The retriever ranks legal source chunks with BM25.
3. The answer generator only uses the top retrieved source.
4. The system prints the citation/source label at the end.
5. The retrieved source list makes the grounding auditable.

## Metrics To Mention

Full benchmark results from the project:

| Metric | Result |
|---|---:|
| BM25 Recall@10 | 0.975 |
| QA Token F1 | 0.799 |
| Top-5 Source Hit | 0.908 |
| Citation Accuracy | 0.813 |
| Judge Faithfulness | 0.858 |

## If Asked About Optimization

- Dense multilingual MiniLM underperformed BM25.
- Full CPU embedding fine-tuning was run, but the naive triplet setup degraded dense retrieval, so BM25 stayed in the demo.
- The pretrained general-domain reranker hurt ranking, but CPU legal-domain fine-tuning improved it strongly.
- The fine-tuned reranker still did not beat direct BM25 on the full benchmark.
- LLM/SFT smoke training was run with FLAN-T5-small, but generated citation quality was too weak for the live demo.
- The optional `local_hf` demo mode loads the fine-tuned FLAN-T5 checkpoint and applies citation guardrails.
