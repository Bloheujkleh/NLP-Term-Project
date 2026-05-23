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
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_corpus_file(data_dir: Path) -> Path:
    candidates = [
        Path("data") / "corpus.jsonl",
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


def extractive_answer(results: list[tuple[Doc, float]]) -> str:
    if not results:
        return "Bu soru icin kaynak bulunamadi."
    best = results[0][0]
    return f"Kaynaga gore: {best.text}\n\nKaynak: {best.citation}"


def local_hf_answer(question: str, results: list[tuple[Doc, float]], model_name: str, max_new_tokens: int) -> str:
    if not results:
        return "Bu soru icin kaynak bulunamadi."
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    context = "\n\n".join(
        f"[{rank}] Baslik: {doc.title}\nKaynak: {doc.citation}\nMetin: {doc.text}"
        for rank, (doc, _score) in enumerate(results, start=1)
    )
    prompt = (
        "Sen bir Turk hukuku RAG asistanisin. Yalnizca verilen kaynaklara dayanarak "
        "kisa ve dogru cevap ver. Kaynakta olmayan bilgiyi uretme.\n\n"
        f"Kaynaklar:\n{context}\n\n"
        f"Soru: {question}\n\n"
        "Cevap:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    generated = re.sub(r"^\s*Soru:\s*", "", generated, flags=re.IGNORECASE)
    generated = re.split(r"\[\d+\]\s*Baslik:|\n\s*Baslik:|\n\s*Metin:", generated, maxsplit=1)[0].strip()
    generated = re.sub(r"\s+", " ", generated).strip()
    if len(generated.split()) < 5:
        generated = results[0][0].text
    if "Kaynak:" not in generated:
        generated = f"{generated}\n\nKaynak: {results[0][0].citation}"
    return generated


def answer(
    question: str,
    results: list[tuple[Doc, float]],
    answer_mode: str,
    generation_model: str | None,
    max_new_tokens: int,
) -> str:
    if answer_mode == "extractive":
        return extractive_answer(results)
    if not generation_model:
        raise ValueError("--generation-model is required when --answer-mode local_hf")
    return local_hf_answer(question, results, generation_model, max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Turkish legal RAG demo")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--corpus-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", default=None)
    parser.add_argument("--answer-mode", choices=["extractive", "local_hf"], default="extractive")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    corpus_file = args.corpus_file or resolve_corpus_file(args.data_dir)
    docs = load_docs(corpus_file, limit=args.limit)
    retriever = SimpleBM25(docs)
    print(f"Loaded {len(docs)} documents from {corpus_file}")
    print(f"Answer mode: {args.answer_mode}")
    if args.generation_model:
        print(f"Generation model: {args.generation_model}")

    questions = [args.question] if args.question else SAMPLE_QUESTIONS
    for question in questions:
        print("\n" + "=" * 88)
        print("Soru:", question)
        results = retriever.search(question, top_k=args.top_k)
        print("\nCevap:\n" + answer(question, results, args.answer_mode, args.generation_model, args.max_new_tokens))
        print("\nTop kaynaklar:")
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"{rank}. {doc.id} | score={score:.3f} | {doc.title}")


if __name__ == "__main__":
    main()
