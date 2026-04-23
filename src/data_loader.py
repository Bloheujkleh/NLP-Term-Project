"""
Load and preprocess real legal datasets into a unified JSONL corpus.

Supported sources:
- Local exported Kaggle files (JSON, JSONL, CSV, TXT)
- HuggingFace dataset: Renicames/turkish-law-chatbot
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from datasets import load_dataset


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces/newlines into a clean single-space text."""
    return re.sub(r"\s+", " ", text).strip()


def first_non_empty(row: Dict, keys: List[str]) -> str:
    """Return first usable value from candidate keys."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        value_str = normalize_whitespace(str(value))
        if value_str:
            return value_str
    return ""


def extract_text_fields(row: Dict) -> tuple[str, str]:
    """
    Map different possible schemas to unified title/text.
    """
    title = first_non_empty(
        row,
        [
            "title",
            "madde_basligi",
            "kanun_adi",
            "document_title",
            "heading",
            "source",
        ],
    )
    # Prefer a single ``text``-like field if present (use before Q+A merge).
    text = first_non_empty(
        row,
        [
            "text",
            "content",
            "body",
            "madde_metni",
            "article_text",
            "context",
        ],
    )
    if text:
        return title, text

    # Turkish QA exports often use Soru + Cevap; English uses question + answer.
    question = first_non_empty(
        row,
        ["question", "soru", "Soru", "query", "input"],
    )
    answer = first_non_empty(
        row,
        ["answer", "cevap", "Cevap", "response", "output"],
    )
    if question or answer:
        merged = normalize_whitespace(f"{question} {answer}".strip())
        if not title and question:
            title = question[:200] + ("…" if len(question) > 200 else "")
        return title, merged

    text = first_non_empty(
        row,
        [
            "answer",
            "cevap",
            "response",
            "Cevap",
        ],
    )
    return title, text


def _row_lower_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map lowercase column names to values (HuggingFace uses e.g. Soru/Cevap)."""
    return {str(k).lower(): v for k, v in row.items()}


def print_hf_dataset_structure(dataset) -> None:
    """Print splits, column names, and a sample row for debugging."""
    print("\n[HuggingFace] Renicames/turkish-law-chatbot — dataset structure:")
    for split_name, split_data in dataset.items():
        cols = split_data.column_names
        print(f"  Split {split_name!r}: columns ({len(cols)}) = {cols}")
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"  Sample row keys: {list(sample.keys())}")


# Keys that usually carry long legal / conversational text (lowercase).
_QUESTION_KEYS: Tuple[str, ...] = (
    "question",
    "soru",
    "query",
    "input",
    "prompt",
    "instruction",
)
_ANSWER_KEYS: Tuple[str, ...] = (
    "answer",
    "cevap",
    "response",
    "output",
    "completion",
    "targets",
)
_TEXT_SINGLE_KEYS: Tuple[str, ...] = (
    "text",
    "content",
    "body",
    "context",
    "document",
    "passage",
    "article",
    "madde_metni",
    "article_text",
    "raw",
    "message",
)
_TITLE_KEYS: Tuple[str, ...] = (
    "title",
    "heading",
    "source",
    "kanun_adi",
    "madde_basligi",
    "document_title",
    "topic",
)
_METADATA_KEYS: Set[str] = {
    "id",
    "idx",
    "index",
    "split",
    "label",
    "labels",
}


def _first_nonempty_from_lower(lower: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        if key not in lower:
            continue
        s = normalize_whitespace(str(lower[key]))
        if s:
            return s
    return ""


def extract_hf_text_and_title(row: Dict[str, Any], split_name: str, row_index: int) -> Tuple[str, str]:
    """
    Build unified (title, text) from one HF row.

    Priority:
    1) If a direct ``text``-like field exists and is non-empty, use it.
    2) If question-like + answer-like fields exist (e.g. Soru + Cevap), merge them.
    3) Otherwise merge remaining string-like fields safely (dedupe parts).
    """
    lower = _row_lower_keys(row)

    # 1) Direct text column
    direct_text = _first_nonempty_from_lower(lower, _TEXT_SINGLE_KEYS)
    if direct_text:
        title = _first_nonempty_from_lower(lower, _TITLE_KEYS)
        if not title:
            title = f"HF {split_name} (text)"
        return title, direct_text

    # 2) Question + answer (Turkish dataset: Soru, Cevap)
    question = _first_nonempty_from_lower(lower, _QUESTION_KEYS)
    answer = _first_nonempty_from_lower(lower, _ANSWER_KEYS)
    if question or answer:
        text = normalize_whitespace(f"{question} {answer}".strip())
        title = _first_nonempty_from_lower(lower, _TITLE_KEYS)
        if not title and question:
            title = question[:200] + ("…" if len(question) > 200 else "")
        elif not title:
            title = f"HF {split_name} QA"
        return title, text

    # 3) Merge all non-metadata string fields (skip duplicates)
    parts: List[str] = []
    seen_lower: Set[str] = set()
    for key in sorted(lower.keys()):
        if key in _METADATA_KEYS:
            continue
        val = lower[key]
        if val is None:
            continue
        s = normalize_whitespace(str(val))
        if not s:
            continue
        s_low = s.lower()
        if s_low in seen_lower:
            continue
        seen_lower.add(s_low)
        parts.append(s)

    merged = normalize_whitespace(" ".join(parts))
    title = _first_nonempty_from_lower(lower, _TITLE_KEYS)
    if not title and parts:
        title = parts[0][:200] + ("…" if len(parts[0]) > 200 else "")
    elif not title:
        title = f"HF {split_name} row {row_index}"
    return title, merged


def _iter_json_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
    elif isinstance(data, dict):
        # Some files wrap records in a top-level key.
        for value in data.values():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row


def _iter_jsonl_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _iter_csv_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def load_kaggle_exported_records(kaggle_dir: Path) -> List[Dict]:
    """
    Read local files from ``data/kaggle_export/`` (JSON, JSONL, CSV, TXT).

    Each record uses a distinct id ``KG_<index>`` (stable order while scanning files).
    Column detection uses ``extract_text_fields`` (text, content, question+answer, Soru+Cevap, etc.).
    """
    records: List[Dict] = []
    kg_index = 0

    supported_files = sorted(
        list(kaggle_dir.rglob("*.json"))
        + list(kaggle_dir.rglob("*.jsonl"))
        + list(kaggle_dir.rglob("*.csv")),
        key=lambda p: str(p),
    )

    for file_path in supported_files:
        try:
            if file_path.suffix.lower() == ".json":
                iterator = _iter_json_records(file_path)
            elif file_path.suffix.lower() == ".jsonl":
                iterator = _iter_jsonl_records(file_path)
            else:
                iterator = _iter_csv_records(file_path)

            for row in iterator:
                title, text = extract_text_fields(row)
                if not text:
                    continue
                rec_id = f"KG_{kg_index}"
                kg_index += 1
                records.append(
                    {
                        "id": rec_id,
                        "title": title or "Kaggle Turkish Law Record",
                        "text": text,
                    }
                )
        except Exception:
            continue

    for txt_path in sorted(kaggle_dir.rglob("*.txt"), key=lambda p: str(p)):
        try:
            with txt_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = normalize_whitespace(line)
                    if not line:
                        continue
                    rec_id = f"KG_{kg_index}"
                    kg_index += 1
                    records.append(
                        {
                            "id": rec_id,
                            "title": txt_path.stem,
                            "text": line,
                        }
                    )
        except Exception:
            continue

    return records


def load_hf_records(exclude_splits: Optional[Set[str]] = None) -> List[Dict]:
    """
    Load HuggingFace dataset: Renicames/turkish-law-chatbot.

    The public JSON uses Turkish keys ``Soru`` (question) and ``Cevap`` (answer);
    we detect columns case-insensitively and combine or merge as needed.
    """
    records: List[Dict] = []
    dataset = load_dataset("Renicames/turkish-law-chatbot")
    print_hf_dataset_structure(dataset)

    for split_name, split_data in dataset.items():
        if exclude_splits is not None and split_name in exclude_splits:
            continue
        for idx, row in enumerate(split_data):
            row_dict = dict(row)
            lower = _row_lower_keys(row_dict)

            title, text = extract_hf_text_and_title(row_dict, split_name, idx)
            text = normalize_whitespace(text)
            if not text:
                continue

            row_id = _first_nonempty_from_lower(lower, ("id", "uuid", "doc_id"))
            if not row_id:
                row_id = str(idx)
            title = normalize_whitespace(title) or f"HF {split_name}"
            records.append(
                {
                    "id": f"HF_{split_name}_{row_id}",
                    "title": title,
                    "text": text,
                }
            )

    return records


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    """Drop duplicate (title, text) pairs."""
    unique: List[Dict] = []
    seen = set()
    for rec in records:
        title = normalize_whitespace(rec.get("title", ""))
        text = normalize_whitespace(rec.get("text", ""))
        rec_id = normalize_whitespace(rec.get("id", ""))
        if not text:
            continue
        key = (title, text)
        if key in seen:
            continue
        seen.add(key)
        unique.append({"id": rec_id or f"DOC_{len(unique)}", "title": title or "Untitled", "text": text})
    return unique


def save_jsonl(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_real_corpus(
    output_path: Path,
    kaggle_dir: Optional[Path] = None,
    include_hf: bool = True,
    hf_exclude_splits: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Merge Kaggle (``KG_*`` ids) + HuggingFace (``HF_*`` ids) into one JSONL corpus.
    """
    kaggle_records: List[Dict] = []
    hf_records: List[Dict] = []

    if kaggle_dir is not None and kaggle_dir.exists():
        kaggle_records = load_kaggle_exported_records(kaggle_dir)
        print(f"[Corpus] Kaggle records loaded: {len(kaggle_records)} (prefix KG_)")

    if include_hf:
        hf_records = load_hf_records(exclude_splits=hf_exclude_splits)
        print(f"[Corpus] HuggingFace records loaded: {len(hf_records)} (prefix HF_)")

    all_records = kaggle_records + hf_records
    cleaned = deduplicate_records(all_records)
    save_jsonl(cleaned, output_path)
    print(f"[Corpus] Merged total (before dedup): {len(all_records)}")
    print(f"[Corpus] Final merged size (after dedup): {len(cleaned)}")
    print(f"[Corpus] Saved to: {output_path}")
    return cleaned


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "real_corpus.jsonl"
    default_kaggle = project_root / "data" / "kaggle_export"

    records = build_real_corpus(
        output_path=default_output,
        kaggle_dir=default_kaggle if default_kaggle.exists() else None,
        include_hf=True,
    )
    print(f"Done. Total valid records: {len(records)} -> {default_output}")
