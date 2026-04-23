"""
Optional lightweight reranking for retrieval results (no heavy cross-encoder).

Combines the existing retrieval score (e.g. RRF) with a simple lexical overlap
between the query and chunk text. Keeps the pipeline runnable everywhere.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Set


def _query_tokens(query: str) -> Set[str]:
    return {t for t in query.lower().split() if len(t) > 1}


def _lexical_overlap(query: str, doc_text: str) -> float:
    """Jaccard-like overlap: |Q ∩ D| / |Q| (0 if query has no tokens)."""
    q = _query_tokens(query)
    if not q:
        return 0.0
    d = {t for t in doc_text.lower().split() if len(t) > 1}
    inter = len(q & d)
    return inter / len(q)


def simple_lexical_rerank(
    results: List[Dict],
    query: str,
    *,
    lex_weight: float = 0.15,
) -> List[Dict]:
    """
    Re-rank by: base_score + lex_weight * lexical_overlap(query, text).

    ``lex_weight`` is small so the original ordering dominates unless ties.
    """
    if not results:
        return []

    base_scores = [float(r.get("score", 0.0)) for r in results]
    max_b = max(base_scores) if base_scores else 0.0
    min_b = min(base_scores) if base_scores else 0.0
    span = max_b - min_b if max_b > min_b else 1.0

    reranked: List[Dict] = []
    for r in results:
        item = deepcopy(r)
        b = float(item.get("score", 0.0))
        norm_b = (b - min_b) / span
        lex = _lexical_overlap(query, item.get("text", ""))
        item["score"] = norm_b + lex_weight * lex
        item["rerank_score"] = item["score"]
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked
