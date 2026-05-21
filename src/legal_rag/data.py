from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("Datasets_Ceng493_legal_rag")


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
    rows = read_jsonl(data_dir / "corpus.jsonl")
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
    return read_json(data_dir / "rag_eval.json")


def load_gold_benchmark(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    return read_json(data_dir / "gold_benchmark.json")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

