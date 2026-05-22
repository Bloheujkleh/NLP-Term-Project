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

## Environment Limitation

The local PyTorch installation is CPU-only and has no CUDA device. Full embedding and cross-encoder reranker fine-tuning were executed on CPU. Embedding fine-tuning degraded dense retrieval compared with the base multilingual MiniLM model. Reranker fine-tuning improved over the pretrained reranker but did not beat the direct BM25 ranking. Large LLM fine-tuning is included as reproducible data/scripts, but final GPU training results were not run locally.

LibreOffice/soffice was not available in this environment, so DOCX-to-PNG visual rendering could not be completed for the report. The DOCX was generated successfully and structurally checked.
