from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/qa_eval_extractive_bm25_full.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qa_error_analysis.md"))
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    answers = data["answers"]
    retrieval_failures = [
        row for row in answers if row["metrics"]["top5_source_hit"] == 0.0
    ]
    top1_failures = [
        row for row in answers
        if row["metrics"]["top1_source_hit"] == 0.0 and row["metrics"]["top5_source_hit"] == 1.0
    ]
    low_f1 = sorted(answers, key=lambda row: row["metrics"]["token_f1"])[: args.max_examples]

    lines = [
        "# QA Error Analysis",
        "",
        "## Summary",
        "",
        f"- Total questions: {len(answers)}",
        f"- Top-5 retrieval failures: {len(retrieval_failures)}",
        f"- Correct source in top-5 but not top-1: {len(top1_failures)}",
        "",
        "## Failure Types",
        "",
        "1. Retrieval failure: the gold source is not present in top-5, so the answer cannot be grounded correctly.",
        "2. Ranking failure: the gold source is in top-5 but not ranked first; the extractive baseline cites the wrong top-1 passage.",
        "3. Answer formatting mismatch: the retrieved source is correct but generated/extractive wording differs from the verified answer.",
        "",
        "## Lowest Token-F1 Examples",
        "",
    ]

    for row in low_f1:
        metrics = row["metrics"]
        lines.extend(
            [
                f"### {row['question_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold source ids: {', '.join(row['gold_source_ids'])}",
                "",
                f"Retrieved ids: {', '.join(row['retrieved_ids'][:5])}",
                "",
                (
                    "Metrics: "
                    f"F1={metrics['token_f1']:.3f}, "
                    f"Top1Hit={metrics['top1_source_hit']:.0f}, "
                    f"Top5Hit={metrics['top5_source_hit']:.0f}, "
                    f"CitationAcc={metrics['citation_label_accuracy']:.0f}"
                ),
                "",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Top-5 retrieval failures: {len(retrieval_failures)}")
    print(f"Ranking failures: {len(top1_failures)}")


if __name__ == "__main__":
    main()

