from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import load_corpus, load_gold_benchmark, write_json
from legal_rag.metrics import (
    citation_label_accuracy,
    exact_match,
    lexical_faithfulness_proxy,
    retrieved_source_hit,
    rouge_l,
    token_f1,
)
from legal_rag.rag import GenerationConfig, build_answer_generator, build_context
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
    parser.add_argument("--output", type=Path, default=Path("outputs/qa_eval.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="bm25")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation-mode", choices=["extractive", "local_hf", "ollama"], default="extractive")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reranker-model", default=None, help="Path or ID of CrossEncoder reranker model")
    parser.add_argument("--rerank-top-n", type=int, default=15, help="Number of candidates to rerank")
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    doc_id_to_citation = {doc.id: doc.citation_label for doc in corpus}
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
    totals = {
        "exact_match": 0.0,
        "token_f1": 0.0,
        "rouge_l": 0.0,
        "top1_source_hit": 0.0,
        "top5_source_hit": 0.0,
        "citation_label_accuracy": 0.0,
        "faithfulness_proxy": 0.0,
    }

    for item in gold:
        retrieve_k = args.rerank_top_n if reranker else args.top_k
        results = retriever.search(item["question"], top_k=retrieve_k)
        if reranker:
            results = reranker.rerank(item["question"], results, top_k=args.top_k)
            
        answer = generator.generate(item["question"], results)
        reference = item["verified_answer"]
        retrieved_ids = [result.doc.id for result in results]
        contexts = [result.doc.text for result in results]
        gold_source_ids = {source["corpus_row_id"] for source in item["gold_sources"]}
        gold_citation_labels = {source.get("citation_label") or doc_id_to_citation.get(source["corpus_row_id"], source["corpus_row_id"]) for source in item["gold_sources"]}

        metrics = {
            "exact_match": exact_match(answer, reference),
            "token_f1": token_f1(answer, reference),
            "rouge_l": rouge_l(answer, reference),
            "top1_source_hit": retrieved_source_hit(retrieved_ids, gold_source_ids, 1),
            "top5_source_hit": retrieved_source_hit(retrieved_ids, gold_source_ids, min(5, args.top_k)),
            "citation_label_accuracy": citation_label_accuracy(answer, gold_citation_labels),
            "faithfulness_proxy": lexical_faithfulness_proxy(answer, contexts),
        }
        for key, value in metrics.items():
            totals[key] += value

        rows.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": answer,
                "reference": reference,
                "retrieved_ids": retrieved_ids,
                "gold_source_ids": sorted(gold_source_ids),
                "gold_citation_labels": sorted(gold_citation_labels),
                "metrics": metrics,
                "context": build_context(results),
            }
        )

    n = len(gold)
    output = {
        "config": {
            "retriever": args.retriever,
            "embedding_model": args.embedding_model if args.retriever != "bm25" else None,
            "top_k": args.top_k,
            "dense_weight": args.dense_weight if args.retriever == "hybrid" else None,
            "generation_mode": args.generation_mode,
            "generation_model": args.generation_model,
            "num_questions": n,
        },
        "summary": {key: value / n for key, value in totals.items()},
        "answers": rows,
    }
    write_json(args.output, output)
    print("QA evaluation complete")
    print(output["config"])
    print(output["summary"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

