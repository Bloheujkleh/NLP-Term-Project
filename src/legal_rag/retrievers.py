from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .data import CorpusDoc, ensure_dir


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class SearchResult:
    doc: CorpusDoc
    score: float


def tokenize_tr(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text)


class DenseRetriever:
    def __init__(
        self,
        docs: list[CorpusDoc],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path = Path("outputs/index"),
        batch_size: int = 64,
    ) -> None:
        self.docs = docs
        self.model_name = model_name
        self.cache_dir = ensure_dir(cache_dir)
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.index = self._load_or_build_index()

    def _cache_prefix(self) -> Path:
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model_name)
        return self.cache_dir / f"dense_{safe_model}_{len(self.docs)}"

    def _load_or_build_index(self) -> faiss.Index:
        prefix = self._cache_prefix()
        index_path = prefix.with_suffix(".faiss")
        ids_path = prefix.with_suffix(".ids.pkl")
        if index_path.exists() and ids_path.exists():
            with ids_path.open("rb") as f:
                cached_ids = pickle.load(f)
            if cached_ids == [doc.id for doc in self.docs]:
                return faiss.read_index(str(index_path))

        texts = [f"{doc.title}\n{doc.text}" for doc in self.docs]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        with ids_path.open("wb") as f:
            pickle.dump([doc.id for doc in self.docs], f)
        return index

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        return [
            SearchResult(self.docs[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def batch_search(self, queries: list[str], top_k: int = 10) -> list[list[SearchResult]]:
        query_embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embeddings, top_k)
        all_results: list[list[SearchResult]] = []
        for query_scores, query_indices in zip(scores, indices):
            all_results.append(
                [
                    SearchResult(self.docs[int(idx)], float(score))
                    for score, idx in zip(query_scores, query_indices)
                    if idx >= 0
                ]
            )
        return all_results


class BM25Retriever:
    def __init__(self, docs: list[CorpusDoc]) -> None:
        self.docs = docs
        tokenized = [tokenize_tr(f"{doc.title} {doc.text}") for doc in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        scores = self.bm25.get_scores(tokenize_tr(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.docs[int(i)], float(scores[int(i)])) for i in top_indices]

    def batch_search(self, queries: list[str], top_k: int = 10) -> list[list[SearchResult]]:
        return [self.search(query, top_k=top_k) for query in queries]


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        dense_weight: float = 0.65,
        candidate_k: int = 50,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.candidate_k = candidate_k

    @staticmethod
    def _minmax(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score:
            return {key: 1.0 for key in scores}
        return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        dense_results = self.dense.search(query, top_k=self.candidate_k)
        bm25_results = self.bm25.search(query, top_k=self.candidate_k)
        return self._combine(dense_results, bm25_results, top_k)

    def batch_search(self, queries: list[str], top_k: int = 10) -> list[list[SearchResult]]:
        dense_batches = self.dense.batch_search(queries, top_k=self.candidate_k)
        bm25_batches = self.bm25.batch_search(queries, top_k=self.candidate_k)
        return [
            self._combine(dense_results, bm25_results, top_k)
            for dense_results, bm25_results in zip(dense_batches, bm25_batches)
        ]

    def _combine(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        doc_by_id = {result.doc.id: result.doc for result in dense_results + bm25_results}
        dense_scores = self._minmax({result.doc.id: result.score for result in dense_results})
        bm25_scores = self._minmax({result.doc.id: result.score for result in bm25_results})

        combined: list[SearchResult] = []
        for doc_id, doc in doc_by_id.items():
            score = self.dense_weight * dense_scores.get(doc_id, 0.0)
            score += (1.0 - self.dense_weight) * bm25_scores.get(doc_id, 0.0)
            combined.append(SearchResult(doc, score))
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:top_k]
