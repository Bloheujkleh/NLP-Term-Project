from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder


def cross_encoder_rerank(
    results: List[Dict],
    query: str,
    model_name_or_path: str,
    *,
    max_length: int = 256,
) -> List[Dict]:
    """
    Re-rank retrieval results with a cross-encoder.

    Each item in ``results`` must contain a ``text`` field.
    """
    if not results:
        return []

    model = CrossEncoder(model_name_or_path, max_length=max_length)
    pairs = [[query, str(r.get("text", ""))] for r in results]
    scores = model.predict(pairs)

    reranked: List[Dict] = []
    for r, s in zip(results, scores):
        item = deepcopy(r)
        item["ce_score"] = float(s)
        item["score"] = float(s)
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


def maybe_cross_encoder_rerank(
    results: List[Dict],
    query: str,
    model_name_or_path: Optional[str],
) -> List[Dict]:
    if not model_name_or_path:
        return results
    return cross_encoder_rerank(results, query, model_name_or_path)
