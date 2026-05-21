from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from .retrievers import SearchResult


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = DEFAULT_RERANKER_MODEL
    batch_size: int = 32
    max_length: int = 512


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        self.model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
        )

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, result.doc.text) for result in candidates]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
        reranked = [
            SearchResult(doc=result.doc, score=float(score))
            for result, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k]

    def batch_rerank(
        self,
        queries: list[str],
        candidate_batches: list[list[SearchResult]],
        top_k: int,
    ) -> list[list[SearchResult]]:
        flat_pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        start = 0
        for query, candidates in zip(queries, candidate_batches):
            flat_pairs.extend((query, result.doc.text) for result in candidates)
            end = start + len(candidates)
            spans.append((start, end))
            start = end

        if not flat_pairs:
            return [[] for _ in candidate_batches]

        flat_scores = self.model.predict(
            flat_pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
        )

        all_results: list[list[SearchResult]] = []
        for candidates, (start, end) in zip(candidate_batches, spans):
            scores = flat_scores[start:end]
            reranked = [
                SearchResult(doc=result.doc, score=float(score))
                for result, score in zip(candidates, scores)
            ]
            reranked.sort(key=lambda result: result.score, reverse=True)
            all_results.append(reranked[:top_k])
        return all_results

