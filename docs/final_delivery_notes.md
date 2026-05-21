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

## Environment Limitation

The local machine is CPU-only and has no CUDA device. Full embedding fine-tuning and cross-encoder reranker fine-tuning are included as reproducible scripts, but final GPU training results were not run locally.

LibreOffice/soffice was not available in this environment, so DOCX-to-PNG visual rendering could not be completed for the report. The DOCX was generated successfully and structurally checked.

