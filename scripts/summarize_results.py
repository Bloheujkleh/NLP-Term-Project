from __future__ import annotations

import json
from pathlib import Path


OUTPUTS = Path("outputs")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    retrieval_files = {
        "BM25": OUTPUTS / "retrieval_eval_bm25_full.json",
        "Dense MiniLM": OUTPUTS / "retrieval_eval_dense_full.json",
        "Hybrid": OUTPUTS / "retrieval_eval_hybrid_full.json",
    }
    qa_file = OUTPUTS / "qa_eval_extractive_bm25_full.json"
    reranker_file = OUTPUTS / "reranker_eval_pretrained_100.json"

    lines = ["# Experiment Summary", "", "## Retrieval", ""]
    lines.append("| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, path in retrieval_files.items():
        summary = read(path)["summary"]
        lines.append(
            f"| {name} | {fmt(summary['recall@5'])} | {fmt(summary['recall@10'])} | "
            f"{fmt(summary['mrr'])} | {fmt(summary['ndcg@10'])} |"
        )

    if reranker_file.exists():
        reranker = read(reranker_file)
        lines.extend(["", "## Reranker", ""])
        lines.append("| System | Recall@5 | Recall@10 | MRR | nDCG@10 |")
        lines.append("|---|---:|---:|---:|---:|")
        first_stage = reranker["first_stage_summary"]
        reranked = reranker["summary"]
        lines.append(
            f"| BM25 first stage | {fmt(first_stage['recall@5'])} | {fmt(first_stage['recall@10'])} | "
            f"{fmt(first_stage['mrr'])} | {fmt(first_stage['ndcg@10'])} |"
        )
        lines.append(
            f"| Pretrained reranker | {fmt(reranked['recall@5'])} | {fmt(reranked['recall@10'])} | "
            f"{fmt(reranked['mrr'])} | {fmt(reranked['ndcg@10'])} |"
        )

    qa = read(qa_file)["summary"]
    lines.extend(["", "## QA", ""])
    lines.append("| System | EM | Token F1 | ROUGE-L | Top-1 Hit | Top-5 Hit | Citation Acc. | Faithfulness |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| BM25 + extractive | {fmt(qa['exact_match'])} | {fmt(qa['token_f1'])} | "
        f"{fmt(qa['rouge_l'])} | {fmt(qa['top1_source_hit'])} | {fmt(qa['top5_source_hit'])} | "
        f"{fmt(qa['citation_label_accuracy'])} | {fmt(qa['faithfulness_proxy'])} |"
    )

    output = OUTPUTS / "experiment_summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

