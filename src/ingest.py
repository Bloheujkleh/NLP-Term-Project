"""
Data loading and simple chunking utilities for a baseline Turkish legal RAG project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str | Path) -> List[Dict]:
    """
    Load a JSONL file where each line is a JSON object.
    """
    records: List[Dict] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def load_corpus(
    project_root: str | Path,
    prefer_real: bool = True,
    real_name: str = "real_corpus.jsonl",
    dummy_name: str = "corpus.jsonl",
) -> List[Dict]:
    """
    Load corpus with fallback:
    1) real corpus if available
    2) dummy corpus otherwise
    """
    root = Path(project_root)
    data_dir = root / "data"
    real_path = data_dir / real_name
    dummy_path = data_dir / dummy_name

    if prefer_real and real_path.exists():
        return load_jsonl(real_path)
    return load_jsonl(dummy_path)


def simple_chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    """
    Split text into fixed-size overlapping character chunks.

    This is intentionally simple for a beginner-friendly baseline:
    - chunk_size: max number of characters in each chunk
    - overlap: how many characters are shared between consecutive chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def build_chunked_corpus(records: List[Dict], chunk_size: int = 220, overlap: int = 40) -> List[Dict]:
    """
    Convert raw legal records into chunk-level documents.

    Each output item contains:
    - chunk_id
    - source_id (original law/document id)
    - title
    - text (chunk content)
    """
    chunked_docs: List[Dict] = []

    for rec in records:
        source_id = str(rec.get("id", "")).strip() or f"DOC_{len(chunked_docs)}"
        title = str(rec.get("title", "")).strip()
        text = str(rec.get("text", "")).strip()
        if not text:
            continue

        chunks = simple_chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            chunked_docs.append(
                {
                    "chunk_id": f"{source_id}_CHUNK_{idx}",
                    "source_id": source_id,
                    "title": title,
                    "text": chunk,
                }
            )

    return chunked_docs
