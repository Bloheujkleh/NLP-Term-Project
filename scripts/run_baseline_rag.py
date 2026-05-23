from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import load_corpus, load_gold_benchmark, write_json
from legal_rag.metrics import exact_match, token_f1
from legal_rag.rag import GenerationConfig, build_answer_generator
from legal_rag.retrievers import BM25Retriever, ChromaDBRetriever, HybridRetriever


def build_retriever(args: argparse.Namespace, corpus):
    if args.retriever == "bm25":
        return BM25Retriever(corpus)
    
    collection_name = "baseline_rag_db"
    if args.embedding_model and ("triplet" in args.embedding_model.lower() or "finetuned" in args.embedding_model.lower()):
        collection_name = "finetuned_rag_db"

    dense = ChromaDBRetriever(
        corpus,
        model_name=args.embedding_model,
        persist_dir=args.index_dir.parent / "chroma_db",
        collection_name=collection_name,
        batch_size=args.batch_size,
    )
    if args.retriever == "dense":
        return dense
    bm25 = BM25Retriever(corpus)
    return HybridRetriever(dense=dense, bm25=bm25, dense_weight=args.dense_weight, candidate_k=args.candidate_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_rag_answers.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation-mode", choices=["extractive", "local_hf", "ollama"], default="extractive")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reranker-model", default=None, help="Path or ID of CrossEncoder reranker model")
    parser.add_argument("--rerank-top-n", type=int, default=15, help="Number of candidates to rerank")
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    gold = load_gold_benchmark(args.data_dir)
    if args.limit:
        gold = gold[: args.limit]

    retriever = build_retriever(args, corpus)
    generator = build_answer_generator(
        GenerationConfig(
            mode=args.generation_mode,
            model_name=args.generation_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )

    reranker = None
    if args.reranker_model:
        from legal_rag.rerankers import CrossEncoderReranker, RerankerConfig
        reranker = CrossEncoderReranker(RerankerConfig(model_name=args.reranker_model))

    rows = []
    total_em = 0.0
    total_f1 = 0.0
    total_citation_hit = 0.0

    for item in gold:
        retrieve_k = args.rerank_top_n if reranker else args.top_k
        results = retriever.search(item["question"], top_k=retrieve_k)
        if reranker:
            results = reranker.rerank(item["question"], results, top_k=args.top_k)
            
        answer = generator.generate(item["question"], results)
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

