from __future__ import annotations

import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts.demo_app import (
    SimpleBM25,
    build_generator,
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
    answer_mode = os.environ.get("ANSWER_MODE", "guarded_causal")
    generation_model = os.environ.get(
        "GENERATION_MODEL",
        "felinabulent/turkish-legal-qwen2-5-0-5b-rag-sft",
    )
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "180"))
    if answer_mode == "extractive":
        generation_model = None

    corpus_file = resolve_corpus_file(data_dir)
    docs = load_docs(corpus_file, limit=limit)
    retriever = SimpleBM25(docs)
    try:
        generator = build_generator(answer_mode, generation_model, max_new_tokens)
    except Exception as exc:
        print(f"Could not load {answer_mode} model ({generation_model}); using extractive fallback: {exc}")
        answer_mode = "extractive"
        generation_model = None
        generator = build_generator(answer_mode, generation_model, max_new_tokens)
    server = ThreadingHTTPServer(
        (host, port),
        build_handler(retriever, generator, answer_mode, generation_model),
    )
    print(f"Loaded {len(docs)} documents from {corpus_file}")
    print(f"Answer mode: {answer_mode}")
    if generation_model:
        print(f"Generation model: {generation_model}")
    print(f"Demo running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
