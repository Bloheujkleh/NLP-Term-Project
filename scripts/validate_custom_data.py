from __future__ import annotations

import argparse
import json
from pathlib import Path


CORPUS_FILES = ["real_corpus.jsonl", "corpus_index.jsonl", "corpus.jsonl"]
BENCHMARK_FILES = ["eval_qa.jsonl", "custom_benchmark.jsonl", "benchmark.jsonl", "eval_qa_150.jsonl"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def find_existing(data_dir: Path, filenames: list[str]) -> Path | None:
    for filename in filenames:
        path = data_dir / filename
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate custom corpus and benchmark files for the RAG system")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--require-benchmark", action="store_true")
    args = parser.parse_args()

    corpus_path = find_existing(args.data_dir, CORPUS_FILES)
    if corpus_path is None:
        raise SystemExit(f"No corpus file found in {args.data_dir}. Expected one of: {', '.join(CORPUS_FILES)}")

    corpus = read_jsonl(corpus_path)
    ids = set()
    for idx, row in enumerate(corpus, start=1):
        for key in ["id", "text"]:
            if key not in row or not str(row[key]).strip():
                raise SystemExit(f"{corpus_path}:{idx} missing required field '{key}'")
        if row["id"] in ids:
            raise SystemExit(f"{corpus_path}:{idx} duplicate id '{row['id']}'")
        ids.add(row["id"])

    benchmark_path = find_existing(args.data_dir, BENCHMARK_FILES)
    benchmark_count = 0
    missing_gold_ids = []
    if benchmark_path:
        benchmark = read_jsonl(benchmark_path)
        benchmark_count = len(benchmark)
        for idx, row in enumerate(benchmark, start=1):
            question = row.get("question") or row.get("query")
            if not question:
                raise SystemExit(f"{benchmark_path}:{idx} missing 'question' or 'query'")
            gold_ids = []
            for key in ["source_id", "gold_source_id"]:
                if row.get(key):
                    gold_ids.append(row[key])
            for key in ["gold_chunk_ids", "relevant_documents", "relevant_doc_ids"]:
                value = row.get(key)
                if isinstance(value, str):
                    gold_ids.append(value)
                elif isinstance(value, list):
                    gold_ids.extend(value)
            for source in row.get("gold_sources", []) or []:
                if source.get("corpus_row_id"):
                    gold_ids.append(source["corpus_row_id"])
            for doc_id in gold_ids:
                if doc_id not in ids:
                    missing_gold_ids.append((idx, doc_id))
    elif args.require_benchmark:
        raise SystemExit(f"No benchmark file found in {args.data_dir}. Expected one of: {', '.join(BENCHMARK_FILES)}")

    if missing_gold_ids:
        preview = ", ".join(f"row {row}: {doc_id}" for row, doc_id in missing_gold_ids[:10])
        raise SystemExit(f"Benchmark references document ids not found in corpus: {preview}")

    print("Custom data validation passed")
    print(f"corpus_file: {corpus_path}")
    print(f"corpus_documents: {len(corpus)}")
    if benchmark_path:
        print(f"benchmark_file: {benchmark_path}")
        print(f"benchmark_questions: {benchmark_count}")
    else:
        print("benchmark_file: not provided")


if __name__ == "__main__":
    main()
