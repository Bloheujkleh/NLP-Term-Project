from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import warnings
import webbrowser
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

warnings.filterwarnings("ignore", "'cgi' is deprecated.*", DeprecationWarning)
import cgi


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


def list_from_mapping(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            return value
    return None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def resolve_corpus_file(data_dir: Path) -> Path:
    candidates = [
        data_dir / "real_corpus.jsonl",
        data_dir / "corpus_index.jsonl",
        data_dir / "corpus.jsonl",
        Path("data") / "real_corpus.jsonl",
        Path("data") / "corpus_index.jsonl",
        Path("data") / "corpus.jsonl",
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


def extract_docx_text(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        text = "".join(texts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_pdf_text(payload: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("PDF destegi icin pypdf kurulmali: pip install pypdf") from exc
    reader = PdfReader(io.BytesIO(payload))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {idx}]\n{page_text.strip()}")
    return "\n\n".join(pages)


def extract_uploaded_text(filename: str, payload: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".json", ".jsonl"}:
        return payload.decode("utf-8", errors="ignore")
    if suffix == ".docx":
        return extract_docx_text(payload)
    if suffix == ".pdf":
        return extract_pdf_text(payload)
    raise ValueError("Desteklenen dosya tipleri: .txt, .md, .csv, .json, .jsonl, .docx, .pdf, .zip")


def docs_from_structured_rows(filename: str, payload: bytes) -> list[Doc]:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in payload.decode("utf-8-sig", errors="ignore").splitlines() if line.strip()]
        elif suffix == ".json":
            loaded = json.loads(payload.decode("utf-8-sig", errors="ignore"))
            rows = loaded if isinstance(loaded, list) else list_from_mapping(
                loaded,
                ("documents", "docs", "corpus", "data", "items", "records", "chunks", "passages"),
            ) or [loaded]
        else:
            return []
    except Exception:
        return []
    docs: list[Doc] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        text = first_text(
            row.get("text"),
            row.get("content"),
            row.get("page_content"),
            row.get("passage"),
            row.get("passage_text"),
            row.get("chunk"),
            row.get("document"),
            row.get("body"),
        )
        if not text:
            continue
        metadata = row.get("metadata") or {}
        doc_id = first_text(
            row.get("id"),
            row.get("source_id"),
            row.get("doc_id"),
            row.get("document_id"),
            row.get("chunk_id"),
            metadata.get("chunk_id"),
            metadata.get("doc_id"),
            f"{Path(filename).stem}_{idx}",
        )
        title = first_text(row.get("title"), metadata.get("title"), metadata.get("category"), Path(filename).name)
        citation = first_text(
            row.get("citation"),
            row.get("citation_label"),
            row.get("source"),
            metadata.get("citation_label"),
            metadata.get("source"),
            f"{title} - {doc_id}",
        )
        docs.append(Doc(doc_id, title, text, citation))
    return docs


def docs_from_uploaded_file(filename: str, payload: bytes) -> list[Doc]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        docs: list[Doc] = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist():
                if info.is_dir() or info.file_size <= 0:
                    continue
                inner_name = Path(info.filename).name
                if not inner_name or inner_name.startswith("."):
                    continue
                inner_suffix = Path(inner_name).suffix.lower()
                if inner_suffix not in {".txt", ".md", ".csv", ".json", ".jsonl", ".docx", ".pdf"}:
                    continue
                inner_payload = archive.read(info)
                docs.extend(docs_from_uploaded_file(inner_name, inner_payload))
        return docs

    structured_docs = docs_from_structured_rows(filename, payload)
    if structured_docs:
        return structured_docs
    text = extract_uploaded_text(filename, payload)
    return chunk_uploaded_text(text, filename)


def chunk_uploaded_text(text: str, filename: str, chunk_size: int = 900, overlap: int = 150) -> list[Doc]:
    text = text.replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[Doc] = []
    safe_stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_").upper() or "UPLOAD"
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if end < len(text):
            split_at = max(chunk.rfind("."), chunk.rfind("?"), chunk.rfind("!"), chunk.rfind("\n"))
            if split_at > int(chunk_size * 0.55):
                chunk = chunk[: split_at + 1].strip()
                end = start + split_at + 1
        if chunk:
            number = len(chunks) + 1
            chunks.append(
                Doc(
                    id=f"{safe_stem}_{number:03d}",
                    title=f"{filename} - chunk {number}",
                    text=chunk,
                    citation=f"Uploaded file: {filename} | chunk {number}",
                )
            )
        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start
    return chunks


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def answer_body_for_metric(answer: str) -> str:
    body = answer.split("Kaynak:", 1)[0]
    body = re.sub(r"^\s*Kaynaga gore:\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"^\s*Kaynağa göre:\s*", "", body, flags=re.IGNORECASE)
    return body.strip()


def contains_normalized(haystack: str, needle: str) -> bool:
    norm_haystack = normalize_text(haystack)
    norm_needle = normalize_text(needle)
    return bool(norm_needle and norm_needle in norm_haystack)


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def first_text(*values) -> str:
    for value in values:
        items = as_list(value)
        if items:
            return items[0]
    return ""


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_benchmark_rows(filename: str, payload: bytes) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                inner_suffix = Path(info.filename).suffix.lower()
                if inner_suffix in {".json", ".jsonl"}:
                    return parse_benchmark_rows(Path(info.filename).name, archive.read(info))
        raise ValueError("Zip icinde benchmark .json veya .jsonl bulunamadi.")
    raw = payload.decode("utf-8-sig", errors="ignore")
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    elif suffix == ".json":
        loaded = json.loads(raw)
        rows = loaded if isinstance(loaded, list) else list_from_mapping(
            loaded,
            ("questions", "qas", "qa", "examples", "benchmark", "data", "items", "records"),
        )
    else:
        raise ValueError("Benchmark dosyasi .json, .jsonl veya .zip olmali.")
    if not isinstance(rows, list):
        raise ValueError("Benchmark list formatinda olmali.")
    return [row for row in rows if isinstance(row, dict)]


def run_custom_benchmark(docs: list[Doc], benchmark_rows: list[dict], generator: AnswerGenerator, limit: int = 200) -> str:
    retriever = SimpleBM25(docs)
    rows = benchmark_rows[:limit]
    if not rows:
        raise ValueError("Benchmark icinde soru bulunamadi.")
    total = 0
    exact = 0
    contains_gold = 0
    f1_sum = 0.0
    top1 = 0
    top5 = 0
    citation = 0
    examples: list[str] = []
    for row in rows:
        question = first_text(row.get("question"), row.get("query"), row.get("soru"))
        gold_answer = first_text(row.get("gold_answer"), row.get("answer"), row.get("answers"), row.get("cevap"))
        gold_sources = (
            as_list(row.get("source_id"))
            or as_list(row.get("source_ids"))
            or as_list(row.get("gold_document"))
            or as_list(row.get("gold_documents"))
            or as_list(row.get("gold_doc_id"))
            or as_list(row.get("gold_doc_ids"))
            or as_list(row.get("relevant_document"))
            or as_list(row.get("relevant_documents"))
            or as_list(row.get("relevant_doc"))
            or as_list(row.get("relevant_docs"))
        )
        if not question:
            continue
        results = retriever.search(question, top_k=5)
        answer = generator.generate(question, results)
        result_ids = [doc.id for doc, _score in results]
        result_citations = [doc.citation for doc, _score in results]
        answer_body = answer_body_for_metric(answer)
        f1 = token_f1(answer_body, gold_answer) if gold_answer else 0.0
        f1_sum += f1
        if gold_answer and normalize_text(answer_body) == normalize_text(gold_answer):
            exact += 1
        if gold_answer and contains_normalized(answer_body, gold_answer):
            contains_gold += 1
        if gold_sources:
            joined_first = f"{result_ids[0]} {result_citations[0]}" if result_ids else ""
            joined_all = " ".join(result_ids + result_citations)
            if result_ids and any(contains_normalized(joined_first, source) for source in gold_sources):
                top1 += 1
            if any(contains_normalized(joined_all, source) for source in gold_sources):
                top5 += 1
            if any(contains_normalized(answer, source) for source in gold_sources):
                citation += 1
        total += 1
        if len(examples) < 3:
            examples.append(
                f"Q: {question}\nA: {answer[:500]}\nTop source: {result_citations[0] if result_citations else 'none'}"
            )
    if total == 0:
        raise ValueError("Benchmark icinde gecerli question/query alani bulunamadi.")
    has_gold_answer = any(first_text(row.get("gold_answer"), row.get("answer"), row.get("answers"), row.get("cevap")) for row in rows)
    has_gold_source = any(
        as_list(row.get("source_id"))
        or as_list(row.get("source_ids"))
        or as_list(row.get("gold_document"))
        or as_list(row.get("gold_documents"))
        or as_list(row.get("gold_doc_id"))
        or as_list(row.get("gold_doc_ids"))
        or as_list(row.get("relevant_document"))
        or as_list(row.get("relevant_documents"))
        or as_list(row.get("relevant_doc"))
        or as_list(row.get("relevant_docs"))
        for row in rows
    )
    lines = [
        f"Evaluated questions: {total}",
        f"Exact Match: {exact / total:.3f}" if has_gold_answer else "Exact Match: n/a",
        f"Answer Contains Gold: {contains_gold / total:.3f}" if has_gold_answer else "Answer Contains Gold: n/a",
        f"Token F1: {f1_sum / total:.3f}" if has_gold_answer else "Token F1: n/a",
        f"Top-1 Source Hit: {top1 / total:.3f}" if has_gold_source else "Top-1 Source Hit: n/a",
        f"Top-5 Source Hit: {top5 / total:.3f}" if has_gold_source else "Top-5 Source Hit: n/a",
        f"Citation Accuracy: {citation / total:.3f}" if has_gold_source else "Citation Accuracy: n/a",
        "",
        "Sample outputs:",
        "\n\n".join(examples),
    ]
    return "\n".join(lines)


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


class AnswerGenerator:
    def generate(self, question: str, results: list[tuple[Doc, float]]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(AnswerGenerator):
    @staticmethod
    def citation_with_id(doc: Doc) -> str:
        citation = doc.citation
        if doc.id and not contains_normalized(citation, doc.id):
            citation = f"{citation} - {doc.id}"
        return citation

    @staticmethod
    def clean_excerpt(doc: Doc) -> str:
        text = re.sub(r"\s+", " ", doc.text).strip()
        title = re.sub(r"\s+", " ", doc.title).strip()
        if title and text.lower().startswith(title.lower()):
            text = text[len(title):].lstrip(" :-–—\t")
        if title.endswith("?") and text.lower().startswith(title[:-1].lower()):
            text = text[len(title) - 1 :].lstrip(" ?:-–—\t")
        return text or doc.text

    @staticmethod
    def select_relevant_excerpt(question: str, doc: Doc, max_sentences: int = 2) -> str:
        text = ExtractiveGenerator.clean_excerpt(doc)
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
        if len(sentences) <= 1:
            return text[:1200]
        query_terms = {tok for tok in tokenize(question) if len(tok) > 2}
        scored: list[tuple[int, int, str]] = []
        for idx, sentence in enumerate(sentences):
            sent_tokens = set(tokenize(sentence))
            overlap = len(query_terms & sent_tokens)
            critical_bonus = sum(
                1
                for term in ["ceza", "hapis", "muebbet", "müebbet", "hak", "sure", "süre", "sart", "şart"]
                if term in sent_tokens
            )
            scored.append((overlap + critical_bonus, -idx, sentence))
        selected = sorted(scored, reverse=True)[:max_sentences]
        selected_indices = sorted(-idx for _score, idx, _sentence in selected)
        excerpt = " ".join(sentences[idx] for idx in selected_indices)
        return excerpt[:1200] if excerpt else text[:1200]

    def generate(self, question: str, results: list[tuple[Doc, float]]) -> str:
        if not results:
            return "Bu soru icin kaynak bulunamadi."
        best = results[0][0]
        excerpt = self.select_relevant_excerpt(question, best)
        return f"Kaynaga gore: {excerpt}\n\nKaynak: {self.citation_with_id(best)}"


class LocalHFGenerator(AnswerGenerator):
    def __init__(self, model_name: str, max_new_tokens: int = 128) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    @staticmethod
    def build_prompt(question: str, results: list[tuple[Doc, float]]) -> str:
        context = "\n\n".join(
            f"[{rank}] Baslik: {doc.title}\nKaynak: {doc.citation}\nMetin: {doc.text}"
            for rank, (doc, _score) in enumerate(results, start=1)
        )
        return (
            "Sen bir Turk hukuku RAG asistanisin. Yalnizca verilen kaynaklara dayanarak "
            "kisa ve dogru cevap ver. Kaynakta olmayan bilgiyi uretme.\n\n"
            f"Kaynaklar:\n{context}\n\n"
            f"Soru: {question}\n\n"
            "Cevap:"
        )

    @staticmethod
    def clean_answer(answer: str, fallback_doc: Doc) -> str:
        answer = answer.strip()
        answer = re.sub(r"^\s*Soru:\s*", "", answer, flags=re.IGNORECASE)
        answer = re.split(r"\[\d+\]\s*Baslik:|\n\s*Baslik:|\n\s*Metin:", answer, maxsplit=1)[0].strip()
        answer = re.sub(r"\s+", " ", answer).strip()
        if len(answer.split()) < 5:
            answer = fallback_doc.text
        if "Kaynak:" not in answer:
            answer = f"{answer}\n\nKaynak: {fallback_doc.citation}"
        return answer

    def generate(self, question: str, results: list[tuple[Doc, float]]) -> str:
        if not results:
            return "Bu soru icin kaynak bulunamadi."
        prompt = self.build_prompt(question, results)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return self.clean_answer(answer, results[0][0])


class GuardedCausalGenerator(AnswerGenerator):
    def __init__(self, model_name: str, max_new_tokens: int = 180) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()
        self.fallback = ExtractiveGenerator()

    @staticmethod
    def build_prompt(question: str, results: list[tuple[Doc, float]]) -> str:
        context = "\n\n".join(
            f"[{rank}] Baslik: {doc.title}\nKaynak: {doc.citation}\nMetin: {doc.text[:700]}"
            for rank, (doc, _score) in enumerate(results[:3], start=1)
        )
        return (
            "Sen Turk hukuku icin kaynak kontrollu bir RAG asistanisin.\n"
            "Kurallar:\n"
            "1. Yalnizca verilen KAYNAKLAR bolumundeki bilgilere dayan.\n"
            "2. Kaynakta olmayan bilgi, madde numarasi veya ceza uretme.\n"
            "3. Kaynaklar yeterli degilse 'Yuklenen kaynaklarda bu soruya yeterli cevap yok.' de.\n"
            "4. Cevabi en fazla 1-2 cumlelik Turkce ve net yaz. Ornek, genel yorum veya ek aciklama ekleme.\n"
            "5. En sonda mutlaka 'Kaynak: ...' satiri ekle ve en uygun kaynak etiketini kullan.\n\n"
            "Kaynakta soru ile ilgili ceza, sure, tutar, hak veya sart acikca yaziyorsa "
            "bu kritik bilgiyi cevaba aynen dahil et.\n\n"
            f"KAYNAKLAR:\n{context}\n\n"
            f"SORU: {question}\n\n"
            "CEVAP:"
        )

    @staticmethod
    def is_supported(answer: str, results: list[tuple[Doc, float]]) -> bool:
        if not answer or len(answer.split()) < 8 or "Kaynak:" not in answer:
            return False
        answer_part = answer.split("Kaynak:", 1)[0].strip()
        if len(tokenize(answer_part)) < 6:
            return False
        if re.search(r"\bSoru\s*:", answer_part, flags=re.IGNORECASE):
            return False
        top_context_tokens = set(tokenize(results[0][0].text))
        answer_token_set = set(tokenize(answer))
        critical_terms = {
            "muebbet",
            "müebbet",
            "hapis",
            "cezasi",
            "cezası",
            "tazminat",
            "sure",
            "süre",
            "gun",
            "gün",
            "ay",
            "yil",
            "yıl",
            "hak",
            "sart",
            "şart",
        }
        missing_critical = [
            term for term in critical_terms if term in top_context_tokens and term not in answer_token_set
        ]
        if len(missing_critical) >= 2:
            return False
        context_tokens = set(tokenize(" ".join(doc.text for doc, _score in results[:3])))
        answer_tokens = [tok for tok in tokenize(answer) if len(tok) > 3]
        if not answer_tokens:
            return False
        overlap = sum(1 for tok in answer_tokens if tok in context_tokens) / len(answer_tokens)
        return overlap >= 0.55

    @staticmethod
    def clean_generated_text(text: str, fallback_doc: Doc) -> str:
        text = text.strip()
        if "CEVAP:" in text:
            text = text.split("CEVAP:", 1)[-1].strip()
        text = re.split(r"\n\s*KAYNAKLAR:|\n\s*SORU:", text, maxsplit=1)[0].strip()
        text = re.sub(r"\s+", " ", text).strip()
        answer_part = text.split("Kaynak:", 1)[0].strip()
        answer_part = re.split(r"\bSoru\s*:", answer_part, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer_part) if part.strip()]
        if sentences:
            text = " ".join(sentences[:2])
        text = text.replace(" Kaynak:", "\n\nKaynak:")
        if "Kaynak:" not in text:
            text = f"{text}\n\nKaynak: {ExtractiveGenerator.citation_with_id(fallback_doc)}"
        return text

    def generate(self, question: str, results: list[tuple[Doc, float]]) -> str:
        if not results:
            return "Bu soru icin kaynak bulunamadi."
        try:
            prompt = self.build_prompt(question, results)
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "Kaynak disina cikmayan Turkce RAG asistani."},
                    {"role": "user", "content": prompt},
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = self.clean_generated_text(answer, results[0][0])
            if self.is_supported(answer, results):
                return answer
        except Exception as exc:
            print(f"Guarded causal generation failed, using extractive fallback: {exc}")
        return self.fallback.generate(question, results)


def build_generator(answer_mode: str, generation_model: str | None, max_new_tokens: int) -> AnswerGenerator:
    if answer_mode == "extractive":
        return ExtractiveGenerator()
    if not generation_model:
        raise ValueError("--generation-model is required for local model answer modes")
    if answer_mode == "local_hf":
        return LocalHFGenerator(generation_model, max_new_tokens=max_new_tokens)
    if answer_mode == "guarded_causal":
        return GuardedCausalGenerator(generation_model, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown answer mode: {answer_mode}")


def page(
    question: str = "",
    answer: str = "",
    results: list[tuple[Doc, float]] | None = None,
    answer_mode: str = "extractive",
    generation_model: str | None = None,
    selected_runtime_mode: str = "extractive",
    upload_question: str = "",
    upload_answer: str = "",
    upload_results: list[tuple[Doc, float]] | None = None,
    upload_message: str = "",
    eval_report: str = "",
) -> bytes:
    results = results or []
    upload_results = upload_results or []
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
    upload_source_cards = "".join(
        f"""
        <article class="source">
          <div class="rank">#{rank} | score {score:.3f}</div>
          <h3>{html.escape(doc.title)}</h3>
          <p>{html.escape(doc.text[:900])}</p>
          <code>{html.escape(doc.citation)}</code>
        </article>
        """
        for rank, (doc, score) in enumerate(upload_results, start=1)
    )
    model_line = f" | <strong>Optional LLM:</strong> {html.escape(generation_model)}" if generation_model else ""
    extractive_selected = "selected" if selected_runtime_mode != "guarded_causal" else ""
    llm_selected = "selected" if selected_runtime_mode == "guarded_causal" else ""
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
    textarea, select {{ width: 100%; font-size: 17px; padding: 12px; box-sizing: border-box; }}
    textarea {{ min-height: 88px; }}
    .actions {{ display: flex; gap: 10px; align-items: center; margin-top: 10px; }}
    button {{ border: 1px solid #2e74b5; background: #2e74b5; color: white; padding: 10px 14px; border-radius: 6px; cursor: pointer; }}
    .samples button {{ margin: 6px 6px 0 0; background: white; color: #2e74b5; }}
    input[type=file] {{ display: block; margin: 10px 0; }}
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
    <p>BM25 retrieval + source-grounded answer + citations</p>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><strong>0.975</strong>Recall@10 BM25</div>
      <div class="metric"><strong>0.799</strong>QA Token F1</div>
      <div class="metric"><strong>0.908</strong>Top-5 source hit</div>
      <div class="metric"><strong>0.813</strong>Citation accuracy</div>
    </section>
    <section class="panel"><strong>Default answer mode:</strong> {html.escape(answer_mode)}{model_line}</section>
    <section class="panel">
      <form method="post" action="/ask">
        <label for="question"><strong>Legal question</strong></label>
        <textarea id="question" name="question">{html.escape(question)}</textarea>
        <label for="runtime_mode"><strong>Answer engine</strong></label>
        <select id="runtime_mode" name="runtime_mode">
          <option value="extractive" {extractive_selected}>Fast source-grounded extractive</option>
          <option value="guarded_causal" {llm_selected}>Fine-tuned Qwen LLM (slower, guarded fallback)</option>
        </select>
        <div class="actions"><button type="submit">Ask RAG</button></div>
        <div class="samples">{sample_buttons}</div>
      </form>
    </section>
    {f'<section class="panel"><h2>Answer</h2><pre>{html.escape(answer)}</pre></section>' if answer else ''}
    {f'<section><h2>Retrieved sources</h2>{source_cards}</section>' if results else ''}
    <section class="panel">
      <h2>Custom Document Test</h2>
      <form method="post" action="/upload_ask" enctype="multipart/form-data">
        <label for="custom_file"><strong>Upload a document</strong></label>
        <input id="custom_file" name="custom_file" type="file" accept=".zip,.txt,.md,.csv,.json,.jsonl,.docx,.pdf">
        <label for="upload_question"><strong>Question for uploaded document</strong></label>
        <textarea id="upload_question" name="upload_question">{html.escape(upload_question)}</textarea>
        <label for="upload_runtime_mode"><strong>Answer engine</strong></label>
        <select id="upload_runtime_mode" name="runtime_mode">
          <option value="extractive" {extractive_selected}>Fast source-grounded extractive</option>
          <option value="guarded_causal" {llm_selected}>Fine-tuned Qwen LLM (slower, guarded fallback)</option>
        </select>
        <div class="actions"><button type="submit">Ask Uploaded Document</button></div>
      </form>
      {f'<p><strong>{html.escape(upload_message)}</strong></p>' if upload_message else ''}
    </section>
    {f'<section class="panel"><h2>Uploaded Document Answer</h2><pre>{html.escape(upload_answer)}</pre></section>' if upload_answer else ''}
    {f'<section><h2>Uploaded document sources</h2>{upload_source_cards}</section>' if upload_results else ''}
    <section class="panel">
      <h2>Custom Benchmark Evaluation</h2>
      <form method="post" action="/eval_upload" enctype="multipart/form-data">
        <label for="eval_corpus"><strong>Upload corpus / document collection</strong></label>
        <input id="eval_corpus" name="eval_corpus" type="file" accept=".zip,.txt,.md,.csv,.json,.jsonl,.docx,.pdf">
        <label for="eval_benchmark"><strong>Upload benchmark</strong> (.json/.jsonl with question, gold_answer, source_id)</label>
        <input id="eval_benchmark" name="eval_benchmark" type="file" accept=".json,.jsonl,.zip">
        <div class="actions"><button type="submit">Run Custom Benchmark</button></div>
      </form>
      {f'<h3>Custom Benchmark Results</h3><pre>{html.escape(eval_report)}</pre>' if eval_report else ''}
    </section>
  </main>
</body>
</html>"""
    return body.encode("utf-8")


def build_handler(
    retriever: SimpleBM25,
    generator: AnswerGenerator,
    answer_mode: str,
    generation_model: str | None,
    llm_generator: AnswerGenerator | None = None,
):
    class DemoHandler(BaseHTTPRequestHandler):
        @staticmethod
        def choose_generator(runtime_mode: str) -> tuple[str, AnswerGenerator]:
            if runtime_mode == "guarded_causal" and llm_generator is not None:
                return "guarded_causal", llm_generator
            return "extractive", generator

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page(answer_mode=answer_mode, generation_model=generation_model))

        def do_POST(self) -> None:
            if self.path == "/upload_ask":
                self.handle_upload_ask()
                return
            if self.path == "/eval_upload":
                self.handle_eval_upload()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            parsed = parse_qs(payload)
            question = parsed.get("question", [""])[0].strip()
            runtime_mode = parsed.get("runtime_mode", ["extractive"])[0].strip()
            selected_mode, active_generator = self.choose_generator(runtime_mode)
            results = retriever.search(question, top_k=5) if question else []
            answer = active_generator.generate(question, results) if question else ""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                page(
                    question,
                    answer,
                    results,
                    answer_mode=answer_mode,
                    generation_model=generation_model,
                    selected_runtime_mode=selected_mode,
                )
            )

        def handle_upload_ask(self) -> None:
            upload_question = ""
            upload_answer = ""
            upload_results: list[tuple[Doc, float]] = []
            upload_message = ""
            selected_mode = "extractive"
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(content_length)
                form = cgi.FieldStorage(
                    fp=io.BytesIO(payload),
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": str(len(payload)),
                    },
                )
                upload_question = str(form.getfirst("upload_question", "")).strip()
                runtime_mode = str(form.getfirst("runtime_mode", "extractive")).strip()
                selected_mode, active_generator = self.choose_generator(runtime_mode)
                file_item = form["custom_file"] if "custom_file" in form else None
                if not upload_question:
                    raise ValueError("Lutfen yuklenen dokuman icin bir soru yazin.")
                if file_item is None or not getattr(file_item, "filename", ""):
                    raise ValueError("Lutfen .txt, .md, .docx, .pdf, .jsonl veya .zip dosyasi secin.")
                filename = Path(file_item.filename).name
                payload = file_item.file.read()
                upload_docs = docs_from_uploaded_file(filename, payload)
                if not upload_docs:
                    raise ValueError("Yuklenen dosyadan okunabilir metin cikarilamadi.")
                custom_retriever = SimpleBM25(upload_docs)
                upload_results = custom_retriever.search(upload_question, top_k=5)
                upload_answer = active_generator.generate(upload_question, upload_results)
                upload_message = f"{filename} indexed with {len(upload_docs)} chunks."
            except Exception as exc:
                upload_message = f"Upload error: {exc}"
                selected_mode = "extractive"

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                page(
                    answer_mode=answer_mode,
                    generation_model=generation_model,
                    selected_runtime_mode=selected_mode,
                    upload_question=upload_question,
                    upload_answer=upload_answer,
                    upload_results=upload_results,
                    upload_message=upload_message,
                )
            )

        def handle_eval_upload(self) -> None:
            eval_report = ""
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(content_length)
                form = cgi.FieldStorage(
                    fp=io.BytesIO(payload),
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": str(len(payload)),
                    },
                )
                corpus_item = form["eval_corpus"] if "eval_corpus" in form else None
                benchmark_item = form["eval_benchmark"] if "eval_benchmark" in form else None
                if corpus_item is None or not getattr(corpus_item, "filename", ""):
                    raise ValueError("Lutfen corpus/document collection dosyasi yukleyin.")
                if benchmark_item is None or not getattr(benchmark_item, "filename", ""):
                    raise ValueError("Lutfen benchmark .json/.jsonl dosyasi yukleyin.")
                corpus_name = Path(corpus_item.filename).name
                benchmark_name = Path(benchmark_item.filename).name
                docs = docs_from_uploaded_file(corpus_name, corpus_item.file.read())
                if not docs:
                    raise ValueError("Corpus dosyasindan okunabilir dokuman cikarilamadi.")
                benchmark_rows = parse_benchmark_rows(benchmark_name, benchmark_item.file.read())
                eval_report = run_custom_benchmark(docs, benchmark_rows, generator)
                eval_report = f"Corpus documents/chunks: {len(docs)}\nBenchmark file: {benchmark_name}\n\n{eval_report}"
            except Exception as exc:
                eval_report = f"Evaluation error: {exc}"

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                page(
                    answer_mode=answer_mode,
                    generation_model=generation_model,
                    eval_report=eval_report,
                )
            )

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
    parser.add_argument("--answer-mode", choices=["extractive", "local_hf", "guarded_causal"], default="extractive")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    corpus_file = args.corpus_file or resolve_corpus_file(args.data_dir)
    docs = load_docs(corpus_file, limit=args.limit)
    retriever = SimpleBM25(docs)
    generator = build_generator(args.answer_mode, args.generation_model, args.max_new_tokens)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(retriever, generator, args.answer_mode, args.generation_model),
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Loaded {len(docs)} documents from {corpus_file}")
    print(f"Answer mode: {args.answer_mode}")
    if args.generation_model:
        print(f"Generation model: {args.generation_model}")
    print(f"Demo running at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
