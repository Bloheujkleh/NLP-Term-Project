from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import load_corpus, load_gold_benchmark, write_json
from legal_rag.metrics import exact_match, token_f1
from legal_rag.rag import extractive_baseline_answer
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
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_rag_answers.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    gold = load_gold_benchmark(args.data_dir)
    if args.limit:
        gold = gold[: args.limit]

    retriever = build_retriever(args, corpus)
    rows = []
    total_em = 0.0
    total_f1 = 0.0
    total_citation_hit = 0.0

    for item in gold:
        results = retriever.search(item["question"], top_k=args.top_k)
        answer = extractive_baseline_answer(item["question"], results)
        reference = item["verified_answer"]
        gold_source_ids = {source["corpus_row_id"] for source in item["gold_sources"]}
        retrieved_ids = [result.doc.id for result in results]
        citation_hit = float(bool(set(retrieved_ids[:1]) & gold_source_ids))
        em = exact_match(answer, reference)
        f1 = token_f1(answer, reference)
        total_em += em
        total_f1 += f1
        total_citation_hit += citation_hit
        rows.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": answer,
                "reference": reference,
                "retrieved_ids": retrieved_ids,
                "gold_source_ids": sorted(gold_source_ids),
                "exact_match": em,
                "token_f1": f1,
                "top1_source_hit": citation_hit,
            }
        )

    summary = {
        "num_questions": len(gold),
        "exact_match": total_em / len(gold),
        "token_f1": total_f1 / len(gold),
        "top1_source_hit": total_citation_hit / len(gold),
    }
    output = {
        "config": {
            "retriever": args.retriever,
            "embedding_model": args.embedding_model if args.retriever != "bm25" else None,
            "top_k": args.top_k,
            "dense_weight": args.dense_weight if args.retriever == "hybrid" else None,
        },
        "summary": summary,
        "answers": rows,
    }
    write_json(args.output, output)
    print("Baseline RAG run complete")
    print(output["config"])
    print(summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

