from __future__ import annotations

import argparse
import html
import json
import math
import re
import webbrowser
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs


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
    raise FileNotFoundError("No corpus file found.")


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


def make_answer(results: list[tuple[Doc, float]]) -> str:
    if not results:
        return "Bu soru için kaynak bulunamadı."
    best = results[0][0]
    return f"Kaynağa göre: {best.text}\n\nKaynak: {best.citation}"


def page(question: str = "", answer: str = "", results: list[tuple[Doc, float]] | None = None) -> bytes:
    results = results or []
    sample_buttons = "".join(
        f"<button name='question' value='{html.escape(q)}'>{html.escape(q)}</button>" for q in SAMPLE_QUESTIONS
    )
    source_cards = "".join(
        f"""
        <article class="source">
          <div class="rank">#{rank} | score {score:.3f}</div>
          <h3>{html.escape(doc.title)}</h3>
          <p>{html.escape(doc.text[:900])}</p>
          <code>{html.escape(doc.citation)}</code>
        </article>
        """
        for rank, (doc, score) in enumerate(results, start=1)
    )
    body = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Turkish Legal RAG Demo</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1d2939; background: #f6f8fb; }}
    header {{ background: #0b2545; color: white; padding: 28px 40px; }}
    header h1 {{ margin: 0 0 6px; font-size: 30px; }}
    header p {{ margin: 0; color: #d7e3f4; }}
    main {{ max-width: 1120px; margin: 24px auto; padding: 0 18px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }}
    .metric, .panel, .source {{ background: white; border: 1px solid #d6dee9; border-radius: 8px; padding: 16px; }}
    .metric strong {{ display: block; color: #2e74b5; font-size: 24px; }}
    textarea {{ width: 100%; min-height: 88px; font-size: 17px; padding: 12px; box-sizing: border-box; }}
    .actions {{ display: flex; gap: 10px; align-items: center; margin-top: 10px; }}
    button {{ border: 1px solid #2e74b5; background: #2e74b5; color: white; padding: 10px 14px; border-radius: 6px; cursor: pointer; }}
    .samples button {{ margin: 6px 6px 0 0; background: white; color: #2e74b5; }}
    pre {{ white-space: pre-wrap; font-size: 16px; line-height: 1.45; }}
    .source {{ margin-top: 12px; }}
    .source h3 {{ margin: 6px 0; font-size: 17px; color: #0b2545; }}
    .source p {{ line-height: 1.45; }}
    code {{ color: #667085; }}
    .rank {{ color: #1b7f5a; font-weight: 700; }}
    @media (max-width: 850px) {{ .metrics {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <header>
    <h1>Turkish Legal RAG Demo</h1>
    <p>BM25 retrieval + source-grounded extractive answer + citations</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><strong>0.975</strong>Recall@10 BM25</div>
      <div class="metric"><strong>0.799</strong>QA Token F1</div>
      <div class="metric"><strong>0.908</strong>Top-5 source hit</div>
      <div class="metric"><strong>0.813</strong>Citation accuracy</div>
    </section>
    <section class="panel">
      <form method="post" action="/ask">
        <label for="question"><strong>Legal question</strong></label>
        <textarea id="question" name="question">{html.escape(question)}</textarea>
        <div class="actions"><button type="submit">Ask RAG</button></div>
        <div class="samples">{sample_buttons}</div>
      </form>
    </section>
    {f'<section class="panel"><h2>Answer</h2><pre>{html.escape(answer)}</pre></section>' if answer else ''}
    {f'<section><h2>Retrieved sources</h2>{source_cards}</section>' if results else ''}
  </main>
</body>
</html>"""
    return body.encode("utf-8")


def build_handler(retriever: SimpleBM25):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page())

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            question = parse_qs(payload).get("question", [""])[0].strip()
            results = retriever.search(question, top_k=5) if question else []
            answer = make_answer(results) if question else ""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page(question, answer, results))

        def log_message(self, format: str, *args) -> None:
            return

    return DemoHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser demo for Turkish legal RAG")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--corpus-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    corpus_file = args.corpus_file or resolve_corpus_file(args.data_dir)
    docs = load_docs(corpus_file, limit=args.limit)
    retriever = SimpleBM25(docs)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(retriever))
    url = f"http://{args.host}:{args.port}"
    print(f"Loaded {len(docs)} documents from {corpus_file}")
    print(f"Demo running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
