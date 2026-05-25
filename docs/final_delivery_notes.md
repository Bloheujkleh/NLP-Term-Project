# Final Delivery Notes

## Deliverables

- `deliverables/Turkish_Legal_RAG_Final_Report.docx`
- `deliverables/Turkish_Legal_RAG_Presentation.pptx`

## Verification Performed

- Retrieval scripts were run for BM25, dense, and hybrid retrieval.
- QA evaluation was run on the 240-question gold benchmark.
- Error analysis was generated from the QA evaluation output.
- The final report DOCX was generated and structurally checked with `python-docx`.
- The final presentation PPTX was generated and rendered to 10 PNG previews through artifact-tool.
- The presentation contact sheet was visually inspected.
- The live browser demo was started locally and returned HTTP 200.
- The terminal demo was tested with a sample legal question.
- Full embedding fine-tuning was run on CPU using all 2,059 embedding triples.
- The fine-tuned embedding model was evaluated on the full 1,000-query retrieval benchmark.
- LLM/NLI judge faithfulness evaluation was run on all 240 gold QA examples.
- Full cross-encoder reranker fine-tuning was run on CPU using all 6,752 reranker pairs.
- The fine-tuned reranker was evaluated on the full 1,000-query retrieval benchmark.
- A CPU-safe FLAN-T5 SFT smoke run was completed using `llm.jsonl`.
- Base FLAN-T5-small and SFT FLAN-T5-small were evaluated under the same BM25 top-3 generation pipeline on the same 20-question QA smoke set.
- The fine-tuned local generator was evaluated on a 20-question QA smoke set.
- Controlled ablation notes were added in `docs/controlled_ablation_summary.md`.
- A code evidence map was added in `docs/code_evidence_map.md`.
- The fast local smoke test passed with `python scripts\smoke_test.py --data-dir data`.
- The CLI demo was verified with `python scripts\demo_cli.py --data-dir data --question "Kasten oldurme sucu nedir?"`.
- The optional local generative LLM path was connected to the browser and CLI demos with `--answer-mode local_hf`.
- The local fine-tuned FLAN-T5 demo path was verified from CLI.
- Custom corpus and benchmark support was documented in `docs/submission_custom_evaluation_guide.md`.
- `scripts/validate_custom_data.py` was added to validate instructor-provided data folders.
- `scripts/run_base_vs_finetuned_eval.py` was added to run Base RAG and Fine-tuned RAG on the same custom benchmark.
- The custom sample dataset under `sample_custom_data/` was validated and evaluated successfully.

## Environment Limitation

The local PyTorch installation is CPU-only and has no CUDA device. Full embedding and cross-encoder reranker fine-tuning were executed on CPU. Embedding fine-tuning degraded dense retrieval compared with the base multilingual MiniLM model. Reranker fine-tuning improved over the pretrained reranker but did not beat the direct BM25 ranking. A small CPU FLAN-T5 SFT smoke run improved over the base FLAN-T5-small under the same pipeline, but it was not strong enough for the live demo because citation accuracy remained 0.000. Large LLM fine-tuning should be run on GPU.

LibreOffice/soffice was not available in this environment, so DOCX-to-PNG visual rendering could not be completed for the report. The DOCX was generated successfully and structurally checked.
