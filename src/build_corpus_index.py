"""
Build a retrieval index corpus that excludes HuggingFace ``test`` split rows.

Why:
- The HF ``test`` split is a natural held-out QA evaluation split.
- Excluding ``HF_test_*`` documents from the retrieval index reduces optimistic overlap
  between gold labels and indexed passages.

Output:
  data/corpus_index.jsonl
"""

from __future__ import annotations

from pathlib import Path

from data_loader import build_real_corpus


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "corpus_index.jsonl"
    kaggle_dir = root / "data" / "kaggle_export"

    records = build_real_corpus(
        output_path=out,
        kaggle_dir=kaggle_dir if kaggle_dir.exists() else None,
        include_hf=True,
        hf_exclude_splits={"test"},
    )
    print(f"[IndexCorpus] Wrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
