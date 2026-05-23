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


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


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

        texts = []
        for doc in self.docs:
            combined = f"{doc.title}\n{doc.text}"
            if "e5" in self.model_name.lower():
                texts.append(f"passage: {combined}")
            else:
                texts.append(combined)

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
        if "e5" in self.model_name.lower():
            query_text = f"query: {query}"
        else:
            query_text = query
        query_embedding = self.model.encode(
            [query_text],
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
        texts = []
        for q in queries:
            if "e5" in self.model_name.lower():
                texts.append(f"query: {q}")
            else:
                texts.append(q)
        query_embeddings = self.model.encode(
            texts,
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


class ChromaDBRetriever:
    def __init__(
        self,
        docs: list[CorpusDoc],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        persist_dir: Path = Path("outputs/chroma_db"),
        collection_name: str = "baseline_rag_db",
        batch_size: int = 64,
    ) -> None:
        import chromadb
        self.docs = docs
        self.model_name = model_name
        self.persist_dir = ensure_dir(persist_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.model = SentenceTransformer(model_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        if self.collection.count() == 0 and len(self.docs) > 0:
            self._populate_collection()

    def _cache_path(self) -> Path:
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model_name)
        cache_dir = Path("outputs/index")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"embeddings_{safe_model}_{len(self.docs)}.pkl"

    def _populate_collection(self) -> None:
        cache_file = self._cache_path()
        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}...")
            with cache_file.open("rb") as f:
                embeddings = pickle.load(f)
        else:
            texts_to_encode = []
            for doc in self.docs:
                combined = f"{doc.title}\n{doc.text}"
                if "e5" in self.model_name.lower():
                    texts_to_encode.append(f"passage: {combined}")
                else:
                    texts_to_encode.append(combined)

            print(f"Encoding {len(self.docs)} docs using model '{self.model_name}'...")
            embeddings = self.model.encode(
                texts_to_encode,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
            
            print(f"Caching embeddings to {cache_file}...")
            with cache_file.open("wb") as f:
                pickle.dump(embeddings, f)

        documents = [doc.text for doc in self.docs]
        metadatas = [{
            "title": doc.title,
            "citation_label": doc.citation_label
        } for doc in self.docs]
        ids = [doc.id for doc in self.docs]

        print(f"Indexing {len(documents)} docs into ChromaDB collection '{self.collection_name}'...")
        chunk_size = 2000
        for i in range(0, len(ids), chunk_size):
            self.collection.add(
                embeddings=embeddings[i : i + chunk_size],
                documents=documents[i : i + chunk_size],
                metadatas=metadatas[i : i + chunk_size],
                ids=ids[i : i + chunk_size],
            )
        print("ChromaDB Indexing completed.")

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if "e5" in self.model_name.lower():
            query_text = f"query: {query}"
        else:
            query_text = query
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        search_results: list[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []
        for doc_id, doc_text, metadata, score in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - float(score)
            doc = CorpusDoc(
                id=doc_id,
                text=doc_text,
                title=metadata.get("title", ""),
                citation_label=metadata.get("citation_label", doc_id),
                metadata=metadata
            )
            search_results.append(SearchResult(doc, similarity))
        return search_results

    def batch_search(self, queries: list[str], top_k: int = 10) -> list[list[SearchResult]]:
        texts_to_encode = []
        for query in queries:
            if "e5" in self.model_name.lower():
                texts_to_encode.append(f"query: {query}")
            else:
                texts_to_encode.append(query)
        query_embeddings = self.model.encode(
            texts_to_encode,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )
        all_results: list[list[SearchResult]] = []
        for i in range(len(queries)):
            q_results: list[SearchResult] = []
            if results["ids"] and len(results["ids"]) > i and results["ids"][i]:
                for doc_id, doc_text, metadata, score in zip(
                    results["ids"][i],
                    results["documents"][i],
                    results["metadatas"][i],
                    results["distances"][i],
                ):
                    similarity = 1.0 - float(score)
                    doc = CorpusDoc(
                        id=doc_id,
                        text=doc_text,
                        title=metadata.get("title", ""),
                        citation_label=metadata.get("citation_label", doc_id),
                        metadata=metadata
                    )
                    q_results.append(SearchResult(doc, similarity))
            all_results.append(q_results)
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
        dense: DenseRetriever | ChromaDBRetriever,
        bm25: BM25Retriever,
        dense_weight: float = 0.5,
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        dense_results = self.dense.search(query, top_k=self.candidate_k)
        bm25_results = self.bm25.search(query, top_k=self.candidate_k)
        return self._combine_rrf(dense_results, bm25_results, top_k)

    def batch_search(self, queries: list[str], top_k: int = 10) -> list[list[SearchResult]]:
        dense_batches = self.dense.batch_search(queries, top_k=self.candidate_k)
        bm25_batches = self.bm25.batch_search(queries, top_k=self.candidate_k)
        return [
            self._combine_rrf(dense_results, bm25_results, top_k)
            for dense_results, bm25_results in zip(dense_batches, bm25_batches)
        ]

    def _combine_rrf(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        doc_by_id = {result.doc.id: result.doc for result in dense_results + bm25_results}
        dense_ranks = {result.doc.id: idx + 1 for idx, result in enumerate(dense_results)}
        bm25_ranks = {result.doc.id: idx + 1 for idx, result in enumerate(bm25_results)}

        combined: list[SearchResult] = []
        for doc_id, doc in doc_by_id.items():
            score = 0.0
            if doc_id in dense_ranks:
                score += self.dense_weight * (1.0 / (self.rrf_k + dense_ranks[doc_id]))
            if doc_id in bm25_ranks:
                score += (1.0 - self.dense_weight) * (1.0 / (self.rrf_k + bm25_ranks[doc_id]))
            combined.append(SearchResult(doc, score))

        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:top_k]
