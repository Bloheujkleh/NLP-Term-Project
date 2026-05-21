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

If the browser app fails for any reason, use the CLI fallback:

```bash
python scripts/demo_cli.py --data-dir data --question "Kasten oldurme sucu nedir?"
```

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

## If Asked About Optimization

- Dense multilingual MiniLM underperformed BM25, so embedding fine-tuning is justified.
- A pretrained general-domain reranker hurt performance, so legal-domain reranker fine-tuning is necessary.
- The project includes scripts for embedding and reranker fine-tuning.
- GPU was not available locally, so full fine-tuning is documented as reproducible future/optional work.

