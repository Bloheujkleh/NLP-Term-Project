from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import load_corpus
from legal_rag.metrics import citation_label_accuracy, lexical_faithfulness_proxy, token_f1
from legal_rag.rag import extractive_baseline_answer
from legal_rag.retrievers import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast local health check for the legal RAG pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--question", default="Kasten oldurme sucu nedir?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    corpus = load_corpus(args.data_dir)
    if not corpus:
        raise SystemExit("Smoke test failed: corpus is empty")

    retriever = BM25Retriever(corpus)
    results = retriever.search(args.question, top_k=args.top_k)
    if not results:
        raise SystemExit("Smoke test failed: BM25 returned no results")

    answer = extractive_baseline_answer(args.question, results)
    if "Kaynak:" not in answer:
        raise SystemExit("Smoke test failed: answer does not include a citation line")

    contexts = [result.doc.text for result in results]
    citation_labels = {result.doc.citation_label for result in results[:1]}
    citation_ok = citation_label_accuracy(answer, citation_labels)
    faithfulness = lexical_faithfulness_proxy(answer, contexts)
    self_f1 = token_f1(answer, answer)

    if citation_ok < 1.0:
        raise SystemExit("Smoke test failed: generated citation does not match top source")
    if faithfulness <= 0.0:
        raise SystemExit("Smoke test failed: answer has no lexical support in retrieved context")
    if self_f1 != 1.0:
        raise SystemExit("Smoke test failed: metric sanity check failed")

    print("Smoke test passed")
    print(f"documents: {len(corpus)}")
    print(f"question: {args.question}")
    print(f"top_source: {results[0].doc.id}")
    print(f"citation: {results[0].doc.citation_label}")
    print(f"faithfulness_proxy: {faithfulness:.3f}")


if __name__ == "__main__":
    main()
