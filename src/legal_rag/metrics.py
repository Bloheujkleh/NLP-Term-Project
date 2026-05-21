from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    import math

    score = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    ideal_hits = min(len(gold_ids), k)
    ideal_ids = ["__gold__"] * ideal_hits
    ideal_gold = {"__gold__"}
    # Since duplicate ids collapse in the simple helper, compute ideal directly.
    import math

    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, gold_ids, k) / ideal


def normalize_text(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\sçğıöşüâîû]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), ref_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, pred_token in enumerate(pred_tokens, start=1):
        for j, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def citation_label_accuracy(answer: str, gold_citation_labels: set[str]) -> float:
    if not gold_citation_labels:
        return 0.0
    return float(any(label in answer for label in gold_citation_labels))


def retrieved_source_hit(retrieved_ids: list[str], gold_source_ids: set[str], k: int) -> float:
    return float(bool(set(retrieved_ids[:k]) & gold_source_ids))


def extract_answer_body(answer: str) -> str:
    marker = "\n\nKaynak:"
    if marker in answer:
        return answer.split(marker, 1)[0]
    return answer


def lexical_faithfulness_proxy(answer: str, contexts: list[str]) -> float:
    """A lightweight proxy: answer content tokens covered by retrieved context tokens."""
    answer_tokens = set(normalize_text(extract_answer_body(answer)).split())
    context_tokens = set(normalize_text(" ".join(contexts)).split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)
