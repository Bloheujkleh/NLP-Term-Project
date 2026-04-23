"""
Demo script for full baseline Turkish legal RAG pipeline.

Run:
    python src/demo.py

- USE_LLM = False: stable extractive answers (default).
- USE_LLM = True: try local FLAN-T5, fall back to extractive if output is poor.
- USE_RERANKER = False: optional lightweight lexical rerank after hybrid (RRF).
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path

from cross_encoder_rerank import maybe_cross_encoder_rerank
from data_loader import build_real_corpus
from generator import LocalGenerator
from ingest import build_chunked_corpus, load_corpus, load_jsonl
from metrics import evaluate_answers, evaluate_groundedness, evaluate_retrieval, gold_doc_ids_from_eval_item
from reranker import simple_lexical_rerank
from retriever import BM25Retriever, DenseRetriever, reciprocal_rank_fusion


EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEN_MODEL_NAME = "google/flan-t5-small"

# Optional local fine-tuned artifacts (set env vars or place models under ./models)
DENSE_MODEL_PATH = os.environ.get("DENSE_MODEL_PATH", "").strip() or EMBED_MODEL_NAME
GEN_MODEL_PATH = os.environ.get("GEN_MODEL_PATH", "").strip() or GEN_MODEL_NAME
CROSS_ENCODER_PATH = os.environ.get("CROSS_ENCODER_PATH", "").strip()

DISPLAY_TOP_K = 3
SNIPPET_CHARS = 165

# Presentation-ready defaults (no LLM / no reranker unless you enable below).
USE_LLM = False
USE_RERANKER = False
USE_CROSS_ENCODER = bool(CROSS_ENCODER_PATH)

RETRIEVE_K = 10
EVAL_K = 10


def truncate_snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def print_ranked_list(header: str, results: list) -> None:
    print(f"\n{header}")
    for i, r in enumerate(results[:DISPLAY_TOP_K], start=1):
        title = r.get("title", "") or ""
        t_short = title[:75] + ("…" if len(title) > 75 else "")
        snippet = truncate_snippet(r.get("text", ""))
        print(f"{i}. [{r['source_id']}] {t_short} | skor={r['score']:.4f}")
        print(f"   {snippet}")


def extractive_answer_from_context(top_context: str, max_sentences: int = 2, max_chars: int = 420) -> str:
    t = (top_context or "").strip()
    if not t:
        return "Bağlam yetersiz: ilgili metin bulunamadı."

    parts = re.split(r"(?<=[.!?…])\s+", t, maxsplit=max_sentences)
    out = " ".join(parts[:max_sentences]).strip()
    if not out:
        out = t
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out


def is_poor_llm_output(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 15:
        return True
    low = t.lower()
    if "insufficient" in low and "context" in low and len(t) < 120:
        return True
    return False


def final_answer(
    question: str,
    hybrid_results: list,
    generator: LocalGenerator,
    use_llm: bool,
) -> str:
    contexts = [r["text"] for r in hybrid_results]
    if not contexts:
        return "Bağlam yetersiz: ilgili metin bulunamadı."

    extractive = extractive_answer_from_context(contexts[0])
    if not use_llm:
        return extractive

    try:
        llm_out = generator.generate_answer(question, contexts)
    except Exception:
        return extractive

    if is_poor_llm_output(llm_out):
        return extractive
    return llm_out.strip()


def hybrid_search(bm25: BM25Retriever, dense: DenseRetriever, question: str, k: int) -> list:
    b = bm25.search(question, k=k)
    d = dense.search(question, k=k)
    fused = reciprocal_rank_fusion(b, d, top_n=k)
    if USE_RERANKER:
        fused = simple_lexical_rerank(fused, question)
    if USE_CROSS_ENCODER:
        fused = maybe_cross_encoder_rerank(fused, question, CROSS_ENCODER_PATH)
    return fused


def hybrid_search_with_toggle(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    question: str,
    k: int,
    use_reranker: bool,
    use_cross_encoder: bool,
) -> list:
    b = bm25.search(question, k=k)
    d = dense.search(question, k=k)
    fused = reciprocal_rank_fusion(b, d, top_n=k)
    if use_reranker:
        fused = simple_lexical_rerank(fused, question)
    if use_cross_encoder:
        fused = maybe_cross_encoder_rerank(fused, question, CROSS_ENCODER_PATH)
    return fused


def ensure_working_corpus(project_root: Path) -> Path:
    """
    Prefer a held-out-safe retrieval index:
      data/corpus_index.jsonl  (HF ``test`` split excluded)

    Fallback:
      data/real_corpus.jsonl    """
    index_path = project_root / "data" / "corpus_index.jsonl"
    real_path = project_root / "data" / "real_corpus.jsonl"
    kaggle_dir = project_root / "data" / "kaggle_export"

    if index_path.exists():
        return index_path

    try:
        build_real_corpus(
            output_path=index_path,
            kaggle_dir=kaggle_dir if kaggle_dir.exists() else None,
            include_hf=True,
            hf_exclude_splits={"test"},
        )
        return index_path
    except Exception as exc:
        print(f"[Uyarı] Index corpus oluşturulamadı ({exc}). real_corpus denenecek.")

    try:
        build_real_corpus(
            output_path=real_path,
            kaggle_dir=kaggle_dir if kaggle_dir.exists() else None,
            include_hf=True,
        )
    except Exception as exc:
        print(f"[Uyarı] Birleşik corpus oluşturulamadı ({exc}).")
        print("        Mevcut real_corpus.jsonl veya corpus.jsonl kullanılacak.")

    return real_path


def filter_eval_by_corpus(eval_items: list, corpus_ids: set) -> list:
    """Skip items whose gold source_id(s) are not in the loaded corpus."""
    kept = []
    for item in eval_items:
        gids = gold_doc_ids_from_eval_item(item)
        if not gids:
            continue
        if not any(g in corpus_ids for g in gids):
            continue
        kept.append(item)
    return kept


def load_eval_set(root: Path) -> list:
    """
    Prefer external eval file if exists:
      data/eval_qa_150.jsonl
    Fallback to built-in mini set.
    """
    eval_path = root / "data" / "eval_qa_150.jsonl"
    if eval_path.exists():
        try:
            items = load_jsonl(eval_path)
            if items:
                return items
        except Exception:
            pass
    return UNIFIED_EVAL_SET


# En az 10 soru: her biri soru + altın cevap + geçerli kaynak id (HF_* veya KG_*).
# KG_* maddeleri, Kaggle dışa aktarımı yoksa corpus'ta olmayacağı için güvenle atlanır.
UNIFIED_EVAL_SET: list = [
    {
        "question": "Türkiye'nin devlet şekli nedir?",
        "gold_answer": "Anayasa madde 1'e göre, türkiye'nin devlet şekli cumhuriyettir.",
        "source_id": "HF_train_0",
    },
    {
        "question": "Türkiye'nin resmi dili nedir?",
        "gold_answer": "Anayasa madde 3'e göre, türkiye'nin resmi dili türkçedir.",
        "source_id": "HF_train_20",
    },
    {
        "question": "Türkiye'nin başkenti neresidir?",
        "gold_answer": "Anayasa madde 3, türkiye'nin başkentini ankara olarak belirler.",
        "source_id": "HF_train_21",
    },
    {
        "question": "Anayasa madde 36'ya göre adil yargılanma hakkı nasıl tanımlanır?",
        "gold_answer": (
            "Anayasa madde 36'ya göre, herkes meşru vasıta ve yollardan faydalanmak suretiyle "
            "yargı mercileri önünde davacı veya davalı olarak iddia ve savunma ile adil yargılanma hakkına sahiptir."
        ),
        "source_id": "HF_train_315",
    },
    {
        "question": "Anayasa madde 38 masumiyet karinesini nasıl düzenler?",
        "gold_answer": (
            "Anayasa madde 38, masumiyet karinesini, suçluluğu hükmen sabit oluncaya kadar "
            "kimsenin suçlu sayılamayacağı şeklinde düzenler."
        ),
        "source_id": "HF_train_335",
    },
    {
        "question": "Hırsızlık suçunun unsurları nelerdir?",
        "gold_answer": "Hırsızlık suçunun unsurları, zilyedlik hakkının hukuka aykırı olarak elde edilmesi veya kullanılmasıdır.",
        "source_id": "HF_train_5628",
    },
    {
        "question": "Hırsızlık suçu nasıl işlenir?",
        "gold_answer": (
            "Hırsızlık suçu, bir malı hukuka aykırı olarak zilyedliğini elde etmek veya elinde bulundurmak "
            "amacıyla yasada belirtilen şartlarla hareket edilmesidir."
        ),
        "source_id": "HF_train_5616",
    },
    {
        "question": "Anayasa madde 2'ye göre Türkiye nasıl bir devlettir?",
        "gold_answer": (
            "Anayasa madde 2'ye göre, türkiye cumhuriyeti'nin temel nitelikleri demokratik, laik, sosyal bir hukuk devleti olmasıdır."
        ),
        "source_id": "HF_train_10",
    },
    {
        "question": "Anayasa madde 1 cumhuriyetin hangi tarihte ilan edildiğini belirtir mi?",
        "gold_answer": "Anayasa madde 1, türkiye cumhuriyeti'nin 29 ekim 1923 tarihinde ilan edildiğini belirtir.",
        "source_id": "HF_train_3",
    },
    {
        "question": "Türkiye'nin bayrağı nasıl tanımlanır?",
        "gold_answer": "Anayasa madde 3'te türkiye'nin bayrağı, beyaz ay yıldızlı al bayrak olarak tanımlanır.",
        "source_id": "HF_train_22",
    },
    {
        "question": "Türk ceza kanununda kasten öldürme suçu nedir?",
        "gold_answer": "Bir kişiyi öldürme kastıyla işlenen suçtur.",
        "source_id": "HF_test_7",
    },
    {
        "question": "Olağanüstü hallerde masumiyet karinesinin ihlali yasak mıdır?",
        "gold_answer": "Anayasa madde 15, olağanüstü hallerde masumiyet karinesinin ihlal edilmesinin de yasak olduğunu belirtir.",
        "source_id": "HF_train_134",
    },
    {
        "question": "Anayasa madde 36'ya göre hak arama hürriyetinin sınırlandırılması nasıl düzenlenir?",
        "gold_answer": (
            "Anayasa madde 36, hak arama hürriyetinin sınırlandırılmasını, ancak kanunla ve demokratik toplum düzeninin "
            "gereklerine uygun olarak yapılabileceğini düzenler."
        ),
        "source_id": "HF_train_316",
    },
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    corpus_path = ensure_working_corpus(root)

    if corpus_path.exists():
        raw_records = load_jsonl(corpus_path)
    else:
        raw_records = load_corpus(root, prefer_real=True)

    n_docs = len(raw_records)
    docs = build_chunked_corpus(raw_records, chunk_size=220, overlap=40)
    n_chunks = len(docs)

    print("\n=== Corpus ===")
    print(f"Dosya: {corpus_path}")
    print(f"Kayıt sayısı (doküman): {n_docs}")
    print(f"Parça sayısı (chunk): {n_chunks}")

    if not docs:
        print("[Hata] Ön işleme sonrası doküman yok. Veri dosyalarını kontrol edin.")
        return

    corpus_ids = {str(r.get("id", "")).strip() for r in raw_records if r.get("id")}

    bm25 = BM25Retriever(docs)
    dense = DenseRetriever(docs, model_name=DENSE_MODEL_PATH)
    generator = LocalGenerator(model_name=GEN_MODEL_PATH)

    sample_question = "Suçluluğu hükmen sabit olmadan kişi suçlu kabul edilir mi?"

    bm25_results = bm25.search(sample_question, k=RETRIEVE_K)
    dense_results = dense.search(sample_question, k=RETRIEVE_K)
    hybrid_results = hybrid_search(bm25, dense, sample_question, RETRIEVE_K)

    answer_text = final_answer(sample_question, hybrid_results, generator, USE_LLM)

    print("\n=== Ayarlar ===")
    print(f"Dense model: {DENSE_MODEL_PATH}")
    print(f"Generator model: {GEN_MODEL_PATH}")
    print(f"Cross-encoder: {'Açık (' + CROSS_ENCODER_PATH + ')' if USE_CROSS_ENCODER else 'Kapalı'}")
    print(f"Yerel LLM (FLAN-T5): {'Açık' if USE_LLM else 'Kapalı (öntanımlı, çıkarımsal yanıt)'}")
    print(f"Hibrit sonrası yeniden sıralama: {'Açık' if USE_RERANKER else 'Kapalı'}")

    print("\n=== Soru ===")
    print(sample_question)

    print_ranked_list("=== BM25 Sonuçları ===", bm25_results)
    print_ranked_list("=== Dense Sonuçları ===", dense_results)
    print_ranked_list("=== Hybrid Sonuçları ===", hybrid_results)

    print("\n=== Nihai Yanıt ===")
    print(truncate_snippet(answer_text, max_chars=480))

    full_eval_items = load_eval_set(root)
    retrieval_eval_items = filter_eval_by_corpus(full_eval_items, corpus_ids)
    skipped = len(full_eval_items) - len(retrieval_eval_items)
    if skipped:
        print(f"\n[Değerlendirme] Corpus'ta bulunmayan altın id nedeniyle atlanan örnek: {skipped}")

    # Held-out mode: if all external eval ids are excluded from corpus_index, keep answer eval on full set
    # and use the built-in aligned set for retrieval-side ranking metrics.
    if not retrieval_eval_items and full_eval_items:
        fallback_retrieval_eval = filter_eval_by_corpus(UNIFIED_EVAL_SET, corpus_ids)
        if fallback_retrieval_eval:
            retrieval_eval_items = fallback_retrieval_eval
            print(
                "\n[Not] Held-out eval id'leri index corpus'ta yok. "
                "Retrieval metrikleri için hizalı mini set (UNIFIED_EVAL_SET) kullanıldı."
            )

    def bm25_fn(q, k):
        return bm25.search(q, k=k)

    def dense_fn(q, k):
        return dense.search(q, k=k)

    def hybrid_fn(q, k):
        return hybrid_search(bm25, dense, q, k)

    bm25_metrics = evaluate_retrieval(retrieval_eval_items, bm25_fn, max_k=EVAL_K)
    dense_metrics = evaluate_retrieval(retrieval_eval_items, dense_fn, max_k=EVAL_K)
    hybrid_metrics = evaluate_retrieval(retrieval_eval_items, hybrid_fn, max_k=EVAL_K)

    print("\n=== Retrieval Değerlendirme ===")
    print(
        f"(Recall@5 / Recall@10 / MRR / nDCG@10, üst-{EVAL_K}; "
        f"geçerli örnek: {len(retrieval_eval_items)})"
    )
    print("BM25  ->", bm25_metrics)
    print("Dense ->", dense_metrics)
    print("Hybrid->", hybrid_metrics)

    answer_eval_items = [x for x in full_eval_items if str(x.get("gold_answer", "")).strip()]

    def answer_fn(question: str) -> str:
        h = hybrid_search(bm25, dense, question, RETRIEVE_K)
        return final_answer(question, h, generator, USE_LLM)

    answer_metrics = evaluate_answers(answer_eval_items, answer_fn)
    print("\n=== Cevap Değerlendirme ===")
    print(f"(EM / Token F1 / BLEU1 / ROUGE-L; geçerli örnek: {len(answer_eval_items)})")
    print(answer_metrics)

    def answer_and_context_fn(question: str) -> dict:
        h = hybrid_search(bm25, dense, question, RETRIEVE_K)
        return {
            "answer": final_answer(question, h, generator, USE_LLM),
            "contexts": [x.get("text", "") for x in h],
            "retrieved_source_ids": [x.get("source_id", "") for x in h],
        }

    grounded_metrics = evaluate_groundedness(answer_eval_items, answer_and_context_fn)
    print("\n=== Groundedness Değerlendirme ===")
    print(f"(Faithfulness / CitationAccuracy; geçerli örnek: {len(answer_eval_items)})")
    print(grounded_metrics)

    def run_variant(name: str, use_reranker: bool, use_cross_encoder: bool, use_llm: bool) -> dict:
        def retrieval_fn(q, k):
            return hybrid_search_with_toggle(
                bm25,
                dense,
                q,
                k,
                use_reranker=use_reranker,
                use_cross_encoder=use_cross_encoder,
            )

        def local_answer_fn(q: str) -> str:
            h_local = hybrid_search_with_toggle(
                bm25,
                dense,
                q,
                RETRIEVE_K,
                use_reranker=use_reranker,
                use_cross_encoder=use_cross_encoder,
            )
            return final_answer(q, h_local, generator, use_llm)

        ret_m = evaluate_retrieval(retrieval_eval_items, retrieval_fn, max_k=EVAL_K)
        ans_m = evaluate_answers(answer_eval_items, local_answer_fn)
        return {
            "name": name,
            "Recall@5": ret_m["Recall@5"],
            "Recall@10": ret_m["Recall@10"],
            "MRR": ret_m["MRR"],
            "nDCG@10": ret_m["nDCG@10"],
            "TokenF1": ans_m["TokenF1"],
        }

    print("\n=== Ablation (Otomatik Karşılaştırma) ===")
    variants = [
        run_variant("Baseline(Hybrid, Extractive)", use_reranker=False, use_cross_encoder=False, use_llm=False),
        run_variant("+LexicalReranker", use_reranker=True, use_cross_encoder=False, use_llm=False),
        run_variant(
            "+CrossEncoderReranker",
            use_reranker=False,
            use_cross_encoder=bool(CROSS_ENCODER_PATH),
            use_llm=False,
        ),
        run_variant(
            "+Lexical+CrossEncoder",
            use_reranker=True,
            use_cross_encoder=bool(CROSS_ENCODER_PATH),
            use_llm=False,
        ),
        run_variant("+LLM", use_reranker=False, use_cross_encoder=False, use_llm=True),
        run_variant("+Lexical+LLM", use_reranker=True, use_cross_encoder=False, use_llm=True),
        run_variant(
            "+CrossEncoder+LLM",
            use_reranker=False,
            use_cross_encoder=bool(CROSS_ENCODER_PATH),
            use_llm=True,
        ),
        run_variant(
            "+Lexical+CrossEncoder+LLM",
            use_reranker=True,
            use_cross_encoder=bool(CROSS_ENCODER_PATH),
            use_llm=True,
        ),
    ]
    for v in variants:
        print(
            f"{v['name']}: "
            f"R@5={v['Recall@5']:.3f}, "
            f"R@10={v['Recall@10']:.3f}, "
            f"MRR={v['MRR']:.3f}, "
            f"nDCG@10={v['nDCG@10']:.3f}, "
            f"TokenF1={v['TokenF1']:.3f}"
        )

    results_payload = {
        "settings": {
            "corpus_path": str(corpus_path),
            "EMBED_MODEL_NAME": EMBED_MODEL_NAME,
            "DENSE_MODEL_PATH": DENSE_MODEL_PATH,
            "GEN_MODEL_NAME": GEN_MODEL_NAME,
            "GEN_MODEL_PATH": GEN_MODEL_PATH,
            "CROSS_ENCODER_PATH": CROSS_ENCODER_PATH,
            "USE_LLM": USE_LLM,
            "USE_RERANKER": USE_RERANKER,
            "USE_CROSS_ENCODER": USE_CROSS_ENCODER,
            "RETRIEVE_K": RETRIEVE_K,
            "EVAL_K": EVAL_K,
        },
        "counts": {
            "documents": n_docs,
            "chunks": n_chunks,
            "eval_total": len(full_eval_items),
            "eval_valid": len(answer_eval_items),
            "retrieval_eval_valid": len(retrieval_eval_items),
            "eval_skipped": skipped,
        },
        "retrieval_metrics": {
            "BM25": bm25_metrics,
            "Dense": dense_metrics,
            "Hybrid": hybrid_metrics,
        },
        "answer_metrics": answer_metrics,
        "groundedness_metrics": grounded_metrics,
        "ablation": variants,
    }
    out_path = root / "data" / "results_export.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Çıktı] Sonuçlar kaydedildi: {out_path}")

    print("\n=== Ek örnek sorular (kısa yanıt) ===")
    extras = [
        "Türkiye'nin milli marşı nedir?",
        "Anayasa madde 2'de laik devlet ne demektir?",
        "Zilyetliğin korunması nasıl sağlanır?",
    ]
    for q in extras:
        try:
            h = hybrid_search(bm25, dense, q, RETRIEVE_K)
            ans = final_answer(q, h, generator, USE_LLM)
            print(f"\nS: {q}")
            print(f"C: {truncate_snippet(ans, max_chars=380)}")
        except Exception:
            print(f"\nS: {q}")
            print("C: (yanıt üretilemedi)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[Hata] Demo çalışırken beklenmeyen bir sorun oluştu:")
        traceback.print_exc()
