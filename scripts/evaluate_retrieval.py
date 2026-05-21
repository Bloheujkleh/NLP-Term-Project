from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import load_corpus, load_rag_eval, write_json
from legal_rag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from legal_rag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever


def build_retriever(args: argparse.Namespace, corpus):
    if args.retriever == "bm25":
        return BM25Retriever(corpus)
    dense = DenseRetriever(
        corpus,
        model_name=args.embedding_model,
        cache_dir=args.index_dir,
        batch_size=args.batch_size,
    )
    if args.retriever == "dense":
        return dense
    bm25 = BM25Retriever(corpus)
    return HybridRetriever(dense=dense, bm25=bm25, dense_weight=args.dense_weight, candidate_k=args.candidate_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets_Ceng493_legal_rag"))
    parser.add_argument("--output", type=Path, default=Path("outputs/retrieval_eval.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    eval_rows = load_rag_eval(args.data_dir)
    if args.limit:
        eval_rows = eval_rows[: args.limit]

    retriever = build_retriever(args, corpus)
    per_query = []
    totals = {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0}

    queries = [row["query"] for row in eval_rows]
    result_batches = retriever.batch_search(queries, top_k=max(args.top_k, 10))

    for row, results in zip(eval_rows, result_batches):
        retrieved_ids = [result.doc.id for result in results]
        gold_ids = set(row["gold_chunk_ids"])
        metrics = {
            "recall@5": recall_at_k(retrieved_ids, gold_ids, 5),
            "recall@10": recall_at_k(retrieved_ids, gold_ids, 10),
            "mrr": reciprocal_rank(retrieved_ids, gold_ids),
            "ndcg@10": ndcg_at_k(retrieved_ids, gold_ids, 10),
        }
        for key, value in metrics.items():
            totals[key] += value
        per_query.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "gold_chunk_ids": row["gold_chunk_ids"],
                "retrieved_ids": retrieved_ids[: args.top_k],
                "metrics": metrics,
            }
        )

    summary = {key: value / len(eval_rows) for key, value in totals.items()}
    output = {
        "config": {
            "retriever": args.retriever,
            "embedding_model": args.embedding_model if args.retriever != "bm25" else None,
            "top_k": args.top_k,
            "dense_weight": args.dense_weight if args.retriever == "hybrid" else None,
            "num_queries": len(eval_rows),
        },
        "summary": summary,
        "per_query": per_query,
    }
    write_json(args.output, output)
    print("Retrieval evaluation complete")
    print(output["config"])
    print(summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
