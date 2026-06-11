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
Hosted demo: https://huggingface.co/spaces/felinabulent/turkish-legal-rag-demo
GitHub branch: https://github.com/Bloheujkleh/NLP-Term-Project/tree/custom-submission-support-2026-05-25
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
python scripts/train_cross_encoder_reranker.py --data-dir data --epochs 1 --batch-size 4 --max-length 128 --output-dir outputs/models/legal_cross_encoder_reranker_full_cpu_128

python scripts/run_base_vs_finetuned_eval.py --data-dir data --output-dir outputs/submission_eval_bm25_vs_finetuned_reranker --base-retriever bm25 --finetuned-retriever bm25 --finetuned-reranker-model outputs/models/legal_cross_encoder_reranker_full_cpu_128
```

## Instructor-Facing Claim

The final live demo uses the most reliable measured configuration:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

The hosted Space also includes the fine-tuned Qwen model as an optional guarded
generation path, but the default hosted mode is extractive because the instructor
will test with custom documents and CPU-only hosting. This keeps responses fast,
auditable, and citation-safe.

Fine-tuned embedding, reranker, FLAN-T5, and Qwen generator experiments are
implemented/documented for ablation. The latest benchmark checks show that
fine-tuning does not automatically improve every component, so the final system
is selected based on measured citation reliability and source-hit performance.

## Morning Space Upload Command

Run this in the already-authenticated Colab notebook to publish the latest
deployment folder:

```python
%cd /content/NLP-Term-Project

!git fetch origin custom-submission-support-2026-05-25
!git reset --hard origin/custom-submission-support-2026-05-25

from huggingface_hub import upload_folder

upload_folder(
    folder_path="hf_space",
    repo_id="felinabulent/turkish-legal-rag-demo",
    repo_type="space",
    commit_message="Deploy robust final custom-data RAG app"
)

print("SPACE UPLOAD OK")
```
