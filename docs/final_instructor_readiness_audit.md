# Final Instructor Readiness Audit

## Current Submission Target

- GitHub branch: `custom-submission-support-2026-05-25`
- Hosted demo: `https://huggingface.co/spaces/felinabulent/turkish-legal-rag-demo`
- Space folder: `hf_space/`
- Deployment zip: `hf_space_deployment.zip`

## Instructor Requirements vs Current Project

| Requirement | Current status |
|---|---|
| Working demo | Covered by `hf_space/` and local `scripts/demo_app.py`. |
| Performance metrics | Covered in report/docs and shown in demo header. |
| Instructor custom document collection | Covered by Custom Document Test and Custom Benchmark Evaluation panels. |
| Instructor custom benchmark | Covered by `.json`, `.jsonl`, and `.zip` benchmark upload support. |
| Base RAG vs fine-tuned RAG comparison | Covered in report/docs; branch includes training/evaluation scripts. |
| Ablation for embedding/reranker/LLM | Covered in report/docs; fine-tuned Qwen is available as optional guarded mode. |

## Final Deployment Decision

The hosted Space defaults to:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

Reason: the instructor will test with custom documents on a CPU-hosted Space.
The extractive default is faster, more auditable, and less likely to hallucinate.

The fine-tuned Qwen model remains available as optional guarded mode:

```text
ANSWER_MODE=guarded_causal
GENERATION_MODEL=felinabulent/turkish-legal-qwen2-5-0-5b-rag-sft
```

## Local Verification Completed

The following local checks passed:

- Python syntax check for `hf_space/app.py` and `hf_space/scripts/demo_app.py`.
- HTTP POST `/ask`: returned status `200`, answer, and citation.
- HTTP POST `/upload_ask`: uploaded a text document and returned a grounded answer.
- HTTP POST `/eval_upload`: uploaded custom corpus and benchmark and returned metrics.
- ZIP document collection test: retrieved the correct uploaded source and produced citation accuracy `1.000`.
- Alternative benchmark field names tested: `query`, `answers`, `relevant_documents`, `doc_id`, `page_content`.

## Important Last Step

After pulling the latest branch in Colab, upload `hf_space/` again to the Space.
The latest branch commit contains the robust extractive default and custom
benchmark fixes. Without this re-upload, the public Space may still run the
older slower Qwen-default version.

## Expected Score Risk

Strong points:

- Custom data and custom benchmark requirement is now the strongest part.
- Demo is simple and robust.
- Source ids and citations are exposed in answers and metrics.
- The app supports multiple likely instructor file formats.

Remaining risk:

- The hosted default is extractive rather than live generative LLM. This is a
defensible engineering choice for legal QA and CPU custom-data testing, but it
should be explained as a reliability decision.
- If the instructor strictly expects the hosted default to be the fine-tuned LLM,
the optional `guarded_causal` mode can be enabled, but it is slower on CPU.

Recommended explanation:

```text
We trained and documented fine-tuned LLM experiments, including a Qwen LoRA
model. For the instructor-facing custom-data Space, we defaulted to the measured
reliable source-grounded extractive mode because the evaluation is legal QA on
CPU and must be auditable with citations. The fine-tuned Qwen path is included
as an optional guarded mode, but the default prioritizes stable custom document
testing.
```
