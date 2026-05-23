from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class CorpusDoc:
    id: str
    text: str
    title: str
    citation_label: str
    metadata: dict[str, Any]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_corpus(data_dir: Path = DEFAULT_DATA_DIR) -> list[CorpusDoc]:
    corpus_files = ["real_corpus.jsonl", "corpus_index.jsonl", "corpus.jsonl"]
    path = None
    for filename in corpus_files:
        p = data_dir / filename
        if p.exists() and p.stat().st_size > 10000:
            path = p
            break
    if path is None:
        path = data_dir / "corpus.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No corpus file found in {data_dir}. Tried {corpus_files}")

    rows = read_jsonl(path)
    docs: list[CorpusDoc] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        docs.append(
            CorpusDoc(
                id=row["id"],
                text=row["text"],
                title=row.get("title") or metadata.get("category") or "",
                citation_label=metadata.get("citation_label") or row["id"],
                metadata=metadata,
            )
        )
    return docs


def load_rag_eval(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    qa_jsonl_path = data_dir / "eval_qa_150.jsonl"
    if qa_jsonl_path.exists():
        rows = read_jsonl(qa_jsonl_path)
        normalized = []
        for idx, row in enumerate(rows):
            normalized.append({
                "query_id": f"Q_{idx}",
                "query": row["question"],
                "gold_chunk_ids": [row["source_id"]] if "source_id" in row else row.get("gold_chunk_ids", [])
            })
        return normalized
    
    json_path = data_dir / "rag_eval.json"
    if json_path.exists():
        return read_json(json_path)
    
    raise FileNotFoundError(f"No retrieval evaluation file found in {data_dir}. Expected eval_qa_150.jsonl or rag_eval.json.")


def load_gold_benchmark(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    qa_jsonl_path = data_dir / "eval_qa_150.jsonl"
    if qa_jsonl_path.exists():
        rows = read_jsonl(qa_jsonl_path)
        normalized = []
        for idx, row in enumerate(rows):
            normalized.append({
                "question_id": f"Q_{idx}",
                "question": row["question"],
                "verified_answer": row.get("gold_answer") or row.get("verified_answer") or "",
                "gold_sources": [{"corpus_row_id": row["source_id"]}] if "source_id" in row else row.get("gold_sources", [])
            })
        return normalized

    json_path = data_dir / "gold_benchmark.json"
    if json_path.exists():
        return read_json(json_path)

    raise FileNotFoundError(f"No gold benchmark file found in {data_dir}. Expected eval_qa_150.jsonl or gold_benchmark.json.")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

