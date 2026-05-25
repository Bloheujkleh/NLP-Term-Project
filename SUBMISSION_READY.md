# Submission Ready Notes

## Recommended GitHub Branch

The latest submission-support changes are on:

```text
custom-submission-support-2026-05-25
```

Pull request link:

```text
https://github.com/Bloheujkleh/NLP-Term-Project/pull/new/custom-submission-support-2026-05-25
```

The original `ervagedikli/Turkish-Law-RAG-Optimization` repository could not be pushed to from this machine because GitHub returned a 403 permission error.

## Main Files To Submit

```text
deliverables/Turkish_Legal_RAG_Final_Report.docx
deliverables/Turkish_Legal_RAG_Presentation.pptx
README.md
docs/submission_custom_evaluation_guide.md
docs/submission_evaluation_results.md
docs/controlled_ablation_summary.md
docs/reproducibility_evidence.md
```

## Main Demo Command

```bash
python scripts/demo_app.py --data-dir data
```

Open:

```text
http://127.0.0.1:7860
```

## Custom Data Validation

```bash
python scripts/validate_custom_data.py --data-dir sample_custom_data --require-benchmark
```

## Base RAG vs Fine-Tuned RAG

```bash
python scripts/run_base_vs_finetuned_eval.py --data-dir sample_custom_data --output-dir outputs/sample_submission_eval
```

## Instructor-Facing Claim

The final live demo uses the most reliable measured configuration:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

Fine-tuned embedding, reranker, and FLAN-T5 generator experiments are implemented and documented for ablation. The latest benchmark checks show that fine-tuning does not automatically improve every component, so the final system is selected based on measured citation reliability and source-hit performance.
