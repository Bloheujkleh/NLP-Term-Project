"""
Build a 150-question evaluation set from HuggingFace Turkish legal QA data.

Output format (JSONL):
{
  "question": "...",
  "gold_answer": "...",
  "source_id": "HF_test_123"
}
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def norm(text: str) -> str:
    return " ".join(str(text).split()).strip()


def build_eval_set(output_path: Path, target_size: int = 150) -> int:
    ds = load_dataset("Renicames/turkish-law-chatbot")
    rows = []

    # Held-out split for evaluation (do not index these rows in ``corpus_index.jsonl``).
    split_name = "test"
    if split_name not in ds:
        raise RuntimeError("Expected split 'test' in Renicames/turkish-law-chatbot")

    split = ds[split_name]
    for idx, row in enumerate(split):
        q = norm(row.get("Soru", ""))
        a = norm(row.get("Cevap", ""))
        if not q or not a:
            continue
        rows.append(
            {
                "question": q,
                "gold_answer": a,
                "source_id": f"HF_{split_name}_{idx}",
            }
        )
        if len(rows) >= target_size:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "eval_qa_150.jsonl"
    n = build_eval_set(out, target_size=150)
    print(f"[EvalSet] Wrote {n} items -> {out}")
