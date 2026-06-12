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


class LazyGenerator:
    def __init__(self, answer_mode: str, generation_model: str | None, max_new_tokens: int) -> None:
        self.answer_mode = answer_mode
        self.generation_model = generation_model
        self.max_new_tokens = max_new_tokens
        self._generator = None

    def _load(self):
        if self._generator is not None:
            return self._generator
        try:
            print(f"Loading answer generator: {self.answer_mode} ({self.generation_model})")
            self._generator = build_generator(self.answer_mode, self.generation_model, self.max_new_tokens)
        except Exception as exc:
            print(
                f"Could not load {self.answer_mode} model ({self.generation_model}); "
                f"using extractive fallback: {exc}"
            )
            self.answer_mode = "extractive"
            self.generation_model = None
            self._generator = build_generator("extractive", None, self.max_new_tokens)
        return self._generator

    def generate(self, question, results):
        return self._load().generate(question, results)


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
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "80"))

    corpus_file = resolve_corpus_file(data_dir)
    docs = load_docs(corpus_file, limit=limit)
    retriever = SimpleBM25(docs)
    generator = LazyGenerator("extractive", None, max_new_tokens)
    llm_generator = LazyGenerator("guarded_causal", generation_model, max_new_tokens)
    server = ThreadingHTTPServer(
        (host, port),
        build_handler(
            retriever,
            generator,
            answer_mode,
            generation_model,
            llm_generator=llm_generator,
        ),
    )
    print(f"Loaded {len(docs)} documents from {corpus_file}")
    print(f"Answer mode: {answer_mode}")
    print(f"Optional guarded LLM model: {generation_model}")
    print(f"Demo running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
