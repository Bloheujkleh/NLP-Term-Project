from __future__ import annotations

import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts.demo_app import (
    ExtractiveGenerator,
    SimpleBM25,
    build_handler,
    load_docs,
    resolve_corpus_file,
)


def main() -> None:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    limit_env = os.environ.get("DOC_LIMIT")
    limit = int(limit_env) if limit_env else None
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))

    corpus_file = resolve_corpus_file(data_dir)
    docs = load_docs(corpus_file, limit=limit)
    retriever = SimpleBM25(docs)
    generator = ExtractiveGenerator()
    server = ThreadingHTTPServer(
        (host, port),
        build_handler(retriever, generator, "extractive", None),
    )
    print(f"Loaded {len(docs)} documents from {corpus_file}")
    print(f"Demo running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
