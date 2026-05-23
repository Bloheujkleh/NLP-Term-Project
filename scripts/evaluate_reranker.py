from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import load_corpus, load_rag_eval, write_json
from legal_rag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from legal_rag.rerankers import CrossEncoderReranker, RerankerConfig
from legal_rag.retrievers import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reranker_eval_bm25_cross_encoder.json"))
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reranker-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    eval_rows = load_rag_eval(args.data_dir)
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    first_stage = BM25Retriever(corpus)
    queries = [row["query"] for row in eval_rows]
    candidate_batches = first_stage.batch_search(queries, top_k=args.candidate_k)

    reranker = CrossEncoderReranker(
        RerankerConfig(
            model_name=args.reranker_model,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    )
    reranked_batches = reranker.batch_rerank(queries, candidate_batches, top_k=max(args.top_k, 10))

    per_query = []
    totals = {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}
    first_stage_totals = {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}

    for row, candidates, reranked in zip(eval_rows, candidate_batches, reranked_batches):
        gold_ids = set(row["gold_chunk_ids"])
        candidate_ids = [result.doc.id for result in candidates]
        reranked_ids = [result.doc.id for result in reranked]
        first_stage_metrics = {
            "recall@5": recall_at_k(candidate_ids, gold_ids, 5),
            "recall@10": recall_at_k(candidate_ids, gold_ids, 10),
            "mrr": reciprocal_rank(candidate_ids, gold_ids),
            "ndcg@10": ndcg_at_k(candidate_ids, gold_ids, 10),
        }
        metrics = {
            "recall@5": recall_at_k(reranked_ids, gold_ids, 5),
            "recall@10": recall_at_k(reranked_ids, gold_ids, 10),
            "mrr": reciprocal_rank(reranked_ids, gold_ids),
            "ndcg@10": ndcg_at_k(reranked_ids, gold_ids, 10),
        }
        for key, value in first_stage_metrics.items():
            first_stage_totals[key] += value
        for key, value in metrics.items():
            totals[key] += value
        per_query.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "gold_chunk_ids": row["gold_chunk_ids"],
                "candidate_ids": candidate_ids[: args.candidate_k],
                "reranked_ids": reranked_ids[: args.top_k],
                "first_stage_metrics": first_stage_metrics,
                "metrics": metrics,
            }
        )

    n = len(eval_rows)
    output = {
        "config": {
            "first_stage": "bm25",
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "reranker_model": args.reranker_model,
            "num_queries": n,
        },
        "first_stage_summary": {key: value / n for key, value in first_stage_totals.items()},
        "summary": {key: value / n for key, value in totals.items()},
        "per_query": per_query,
    }
    write_json(args.output, output)
    print("Reranker evaluation complete")
    print(output["config"])
    print("First-stage BM25:", output["first_stage_summary"])
    print("Reranked:", output["summary"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

