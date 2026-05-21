from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SAMPLE_QUESTIONS = [
    "Kasten oldurme sucu nedir?",
    "Adil yargilanma hakki nasil guvence altina alinir?",
    "Evlilik birligi temelinden sarsilirsa ne olur?",
]


@dataclass
class Doc:
    id: str
    title: str
    text: str
    citation: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower())


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_corpus_file(data_dir: Path) -> Path:
    candidates = [
        Path("Datasets_Ceng493_legal_rag") / "corpus.jsonl",
        data_dir / "corpus.jsonl",
        data_dir / "corpus_index.jsonl",
        data_dir / "real_corpus.jsonl",
        Path("data") / "corpus.jsonl",
        Path("data") / "corpus_index.jsonl",
        Path("data") / "real_corpus.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No corpus file found. Expected corpus.jsonl, corpus_index.jsonl, or real_corpus.jsonl.")


def load_docs(corpus_file: Path, limit: int | None = None) -> list[Doc]:
    docs: list[Doc] = []
    for row in iter_jsonl(corpus_file):
        metadata = row.get("metadata") or {}
        doc_id = str(row.get("id") or metadata.get("chunk_id") or len(docs))
        title = str(row.get("title") or metadata.get("category") or "Legal Source")
        text = str(row.get("text") or row.get("content") or "")
        if not text.strip():
            continue
        citation = str(metadata.get("citation_label") or row.get("citation_label") or f"{title} - {doc_id}")
        docs.append(Doc(doc_id, title, text, citation))
        if limit and len(docs) >= limit:
            break
    return docs


class SimpleBM25:
    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self.doc_tokens = [tokenize(f"{doc.title} {doc.text}") for doc in docs]
        self.avgdl = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[Doc, float]]:
        q_terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        k1 = 1.5
        b = 0.75
        for idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                numerator = tf[term] * (k1 + 1)
                denominator = tf[term] + k1 * (1 - b + b * dl / max(self.avgdl, 1))
                score += self.idf.get(term, 0.0) * numerator / denominator
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self.docs[idx], score) for idx, score in scores[:top_k]]


def answer(question: str, results: list[tuple[Doc, float]]) -> str:
    if not results:
        return "Bu soru için kaynak bulunamadı."
    best = results[0][0]
    return f"Kaynağa göre: {best.text}\n\nKaynak: {best.citation}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Turkish legal RAG demo")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--corpus-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    corpus_file = args.corpus_file or resolve_corpus_file(args.data_dir)
    docs = load_docs(corpus_file, limit=args.limit)
    retriever = SimpleBM25(docs)
    print(f"Loaded {len(docs)} documents from {corpus_file}")

    if args.question:
        questions = [args.question]
    else:
        questions = SAMPLE_QUESTIONS

    for question in questions:
        print("\n" + "=" * 88)
        print("Soru:", question)
        results = retriever.search(question, top_k=args.top_k)
        print("\nCevap:\n" + answer(question, results))
        print("\nTop kaynaklar:")
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"{rank}. {doc.id} | score={score:.3f} | {doc.title}")


if __name__ == "__main__":
    main()
