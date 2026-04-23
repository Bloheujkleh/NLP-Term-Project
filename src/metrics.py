"""
Evaluation metrics for retrieval:
- Recall@k
- MRR (Mean Reciprocal Rank)
- nDCG@k

Evaluation metrics for answers:
- Exact Match (EM)
- Token-level F1
- BLEU-1 (lightweight)
- ROUGE-L (lightweight)
- Faithfulness (context overlap heuristic)
- Citation accuracy (retrieval-gold id consistency)
"""

from __future__ import annotations

import math
import re
from typing import Dict, List


def _normalize_doc_id(doc_id: str) -> str:
    """Strip whitespace so JSONL ids match retrieved chunk source_ids reliably."""
    return str(doc_id).strip()


def _unique_preserve_order(ids: List[str]) -> List[str]:
    """Remove duplicates while preserving original ranking order."""
    seen = set()
    out: List[str] = []
    for x in ids:
        nx = _normalize_doc_id(x)
        if nx in seen:
            continue
        seen.add(nx)
        out.append(nx)
    return out


def _gold_ids_from_sample(sample: Dict) -> List[str]:
    """
    Accept ``relevant_source_ids``, ``gold_doc_ids``, or a single ``source_id``
    (HF_* / KG_* corpus ids).
    """
    if "relevant_source_ids" in sample:
        raw = sample["relevant_source_ids"]
    elif "gold_doc_ids" in sample:
        raw = sample["gold_doc_ids"]
    elif "source_id" in sample:
        raw = sample["source_id"]
    else:
        return []
    if raw is None:
        return []
    if isinstance(raw, str):
        return [_normalize_doc_id(raw)]
    if isinstance(raw, (list, tuple)):
        return [_normalize_doc_id(x) for x in raw]
    return [_normalize_doc_id(str(raw))]


def gold_doc_ids_from_eval_item(sample: Dict) -> List[str]:
    """Public helper for demos: get gold document ids from a unified eval item."""
    return _gold_ids_from_sample(sample)


def recall_at_k(retrieved_source_ids: List[str], relevant_source_ids: List[str], k: int) -> float:
    """
    Recall@k for a single query:
    Did we retrieve at least one relevant source in top-k?
    """
    unique_ranked = _unique_preserve_order(retrieved_source_ids)
    top_k = unique_ranked[:k]
    rel = {_normalize_doc_id(x) for x in relevant_source_ids}
    hit = any(doc_id in rel for doc_id in top_k)
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved_source_ids: List[str], relevant_source_ids: List[str]) -> float:
    """
    Reciprocal rank for a single query:
    1 / rank of first relevant item, or 0 if none found.
    """
    rel = {_normalize_doc_id(x) for x in relevant_source_ids}
    unique_ranked = _unique_preserve_order(retrieved_source_ids)
    for idx, doc_id in enumerate(unique_ranked, start=1):
        if _normalize_doc_id(doc_id) in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved_source_ids: List[str], relevant_source_ids: List[str], k: int = 10) -> float:
    """
    Binary nDCG@k:
    - relevant source -> gain 1
    - non-relevant source -> gain 0
    """
    if k <= 0:
        return 0.0
    rel = {_normalize_doc_id(x) for x in relevant_source_ids}
    if not rel:
        return 0.0

    top_k = _unique_preserve_order(retrieved_source_ids)[:k]

    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        gain = 1.0 if doc_id in rel else 0.0
        if gain > 0:
            dcg += gain / math.log2(i + 1)

    ideal_hits = min(k, len(rel))
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0:
        return 0.0
    # Numerical safety + bounded metric definition.
    return min(1.0, dcg / idcg)


def evaluate_retrieval(qa_set: List[Dict], retrieve_fn, max_k: int = 10) -> Dict[str, float]:
    """
    Evaluate a retrieval function over a QA set.

    qa_set item format:
    {
        "question": "...",
        "relevant_source_ids": ["HF_train_0", ...]   # must match corpus record ``id`` / chunk ``source_id``
    }
    (alias: ``gold_doc_ids``)
    """
    recall5_scores: List[float] = []
    recall10_scores: List[float] = []
    rr_scores: List[float] = []
    ndcg10_scores: List[float] = []
    n_eval = 0

    for sample in qa_set:
        question = sample["question"]
        relevant = _gold_ids_from_sample(sample)
        if not relevant:
            continue

        results = retrieve_fn(question, max_k)
        retrieved_ids = [r["source_id"] for r in results]

        recall5_scores.append(recall_at_k(retrieved_ids, relevant, k=5))
        recall10_scores.append(recall_at_k(retrieved_ids, relevant, k=10))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant))
        ndcg10_scores.append(ndcg_at_k(retrieved_ids, relevant, k=10))
        n_eval += 1

    if n_eval == 0:
        return {"Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0, "nDCG@10": 0.0}
    return {
        "Recall@5": sum(recall5_scores) / n_eval,
        "Recall@10": sum(recall10_scores) / n_eval,
        "MRR": sum(rr_scores) / n_eval,
        "nDCG@10": sum(ndcg10_scores) / n_eval,
    }


def normalize_answer(text: str) -> str:
    """Lowercase + remove extra spaces and punctuation for answer comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match_score(prediction: str, gold_answer: str) -> float:
    """Return 1.0 if normalized strings are identical, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold_answer) else 0.0


def token_f1_score(prediction: str, gold_answer: str) -> float:
    """
    Token-level F1 for one prediction-gold pair.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    gold_counts = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    common = 0
    for tok, count in pred_counts.items():
        common += min(count, gold_counts.get(tok, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, gold_answer: str) -> float:
    """Lightweight BLEU-1 with brevity penalty."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    gold_counts = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    overlap = 0
    for tok, cnt in pred_counts.items():
        overlap += min(cnt, gold_counts.get(tok, 0))

    precision = overlap / len(pred_tokens)
    if precision <= 0:
        return 0.0

    bp = 1.0
    if len(pred_tokens) < len(gold_tokens):
        bp = math.exp(1.0 - (len(gold_tokens) / max(1, len(pred_tokens))))
    return bp * precision


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Longest common subsequence length (token level)."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l_score(prediction: str, gold_answer: str) -> float:
    """Token-level ROUGE-L F-score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def faithfulness_score(prediction: str, contexts: List[str]) -> float:
    """
    Heuristic faithfulness:
    ratio of unique answer tokens covered by concatenated contexts.
    """
    pred_tokens = set(normalize_answer(prediction).split())
    if not pred_tokens:
        return 0.0
    ctx_tokens = set(normalize_answer(" ".join(contexts)).split())
    if not ctx_tokens:
        return 0.0
    covered = len(pred_tokens & ctx_tokens)
    return covered / max(1, len(pred_tokens))


def evaluate_answers(answer_set: List[Dict], answer_fn) -> Dict[str, float]:
    """
    Evaluate generated answers on a small QA set.

    answer_set item format:
    {
      "question": "...",
      "gold_answer": "..."
    }
    Skips items with missing or empty ``gold_answer``.
    """
    em_scores: List[float] = []
    f1_scores: List[float] = []
    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    n_eval = 0

    for sample in answer_set:
        question = sample.get("question")
        gold = sample.get("gold_answer")
        if question is None or gold is None:
            continue
        gold = str(gold).strip()
        if not gold:
            continue

        try:
            pred = answer_fn(str(question).strip())
        except Exception:
            pred = ""

        em_scores.append(exact_match_score(pred, gold))
        f1_scores.append(token_f1_score(pred, gold))
        bleu_scores.append(bleu1_score(pred, gold))
        rouge_scores.append(rouge_l_score(pred, gold))
        n_eval += 1

    if n_eval == 0:
        return {"EM": 0.0, "TokenF1": 0.0, "BLEU1": 0.0, "ROUGE-L": 0.0}
    return {
        "EM": sum(em_scores) / n_eval,
        "TokenF1": sum(f1_scores) / n_eval,
        "BLEU1": sum(bleu_scores) / n_eval,
        "ROUGE-L": sum(rouge_scores) / n_eval,
    }


def evaluate_groundedness(answer_set: List[Dict], answer_and_context_fn) -> Dict[str, float]:
    """
    Evaluate source grounding on a QA set.

    ``answer_and_context_fn(question)`` should return:
    {
      "answer": str,
      "contexts": List[str],
      "retrieved_source_ids": List[str]
    }
    """
    faith_scores: List[float] = []
    cite_scores: List[float] = []
    n_eval = 0

    for sample in answer_set:
        question = sample.get("question")
        if not question:
            continue
        relevant = _gold_ids_from_sample(sample)
        try:
            out = answer_and_context_fn(str(question).strip())
        except Exception:
            continue

        answer = str(out.get("answer", ""))
        contexts = list(out.get("contexts", []))
        retrieved_ids = [_normalize_doc_id(x) for x in out.get("retrieved_source_ids", [])]

        faith_scores.append(faithfulness_score(answer, contexts))

        if relevant:
            rel = {_normalize_doc_id(x) for x in relevant}
            hit = any(x in rel for x in retrieved_ids)
            cite_scores.append(1.0 if hit else 0.0)
        else:
            cite_scores.append(0.0)

        n_eval += 1

    if n_eval == 0:
        return {"Faithfulness": 0.0, "CitationAccuracy": 0.0}
    return {
        "Faithfulness": sum(faith_scores) / n_eval,
        "CitationAccuracy": sum(cite_scores) / n_eval,
    }
