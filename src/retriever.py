"""
Retriever implementations:
1) BM25
2) Dense (Sentence Transformers + FAISS)
3) Hybrid (RRF over BM25 and Dense rankings)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize_tr(text: str) -> List[str]:
    """
    Very simple tokenizer for Turkish text.
    Lowercases and splits by whitespace.
    """
    return text.lower().split()


class BM25Retriever:
    """
    BM25 lexical retriever using rank_bm25.
    """

    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.corpus_tokens = [tokenize_tr(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.docs:
            return []
        k = min(k, len(self.docs))
        query_tokens = tokenize_tr(query)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            result = dict(self.docs[idx])
            result["score"] = float(scores[idx])
            results.append(result)
        return results


class DenseRetriever:
    """
    Dense semantic retriever with SentenceTransformer embeddings + FAISS index.
    """

    def __init__(self, docs: List[Dict], model_name: str):
        self.docs = docs
        if not self.docs:
            raise ValueError("DenseRetriever received an empty document list.")
        self.model = SentenceTransformer(model_name)

        # Encode documents and normalize for cosine similarity via inner product.
        doc_texts = [d["text"] for d in docs]
        doc_embeddings = self.model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False)
        doc_embeddings = doc_embeddings.astype("float32")
        faiss.normalize_L2(doc_embeddings)

        dim = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(doc_embeddings)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.docs:
            return []
        k = min(k, len(self.docs))
        query_vec = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            result = dict(self.docs[int(idx)])
            result["score"] = float(score)
            results.append(result)
        return results


def reciprocal_rank_fusion(
    bm25_results: List[Dict], dense_results: List[Dict], k: int = 60, top_n: int = 5
) -> List[Dict]:
    """
    Combine two rankings using Reciprocal Rank Fusion (RRF).

    RRF score for a document:
    sum(1 / (k + rank_i))
    where rank_i starts from 1.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_store: Dict[str, Dict] = {}

    for rank, doc in enumerate(bm25_results, start=1):
        key = doc["chunk_id"]
        rrf_scores[key] += 1.0 / (k + rank)
        doc_store[key] = doc

    for rank, doc in enumerate(dense_results, start=1):
        key = doc["chunk_id"]
        rrf_scores[key] += 1.0 / (k + rank)
        doc_store[key] = doc

    sorted_items: List[Tuple[str, float]] = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for key, score in sorted_items[:top_n]:
        item = dict(doc_store[key])
        item["score"] = float(score)
        final_results.append(item)
    return final_results
