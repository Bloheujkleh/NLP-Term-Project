from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from legal_rag.data import load_corpus, write_json, ensure_dir
from legal_rag.retrievers import BM25Retriever, tokenize_tr


def jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(tokenize_tr(text1))
    words2 = set(tokenize_tr(text2))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives using BM25 retrieval")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/embedding.jsonl"))
    parser.add_argument("--limit", type=int, default=3000, help="Number of triplets to mine")
    parser.add_argument("--overlap-threshold", type=float, default=0.25, 
                        help="Maximum Jaccard similarity between positive and negative passage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dir(args.output.parent)

    print("Loading corpus...")
    corpus = load_corpus(args.data_dir)
    print(f"Corpus loaded: {len(corpus)} documents.")

    print("Initializing BM25 retriever...")
    retriever = BM25Retriever(corpus)

    print("Loading HuggingFace training dataset...")
    ds = load_dataset("Renicames/turkish-law-chatbot")
    train_split = ds["train"]
    print(f"Training dataset size: {len(train_split)}")

    triplets = []
    skipped_no_negative = 0
    skipped_high_overlap = 0

    # Shuffle training rows to get a representative subset
    indices = list(range(len(train_split)))
    random.shuffle(indices)

    print("Mining hard negatives...")
    pbar = tqdm(total=args.limit)
    for idx in indices:
        if len(triplets) >= args.limit:
            break

        row = train_split[idx]
        query = str(row.get("Soru", "")).strip()
        positive = str(row.get("Cevap", "")).strip()

        if not query or not positive:
            continue

        # Retrieve top 15 candidate passages using BM25
        results = retriever.search(query, top_k=15)
        
        # Find a hard negative: a high-scoring passage that is NOT the positive passage
        # and doesn't overlap too much with it (to avoid paraphrased versions of the correct answer)
        hard_negative = None
        for res in results:
            candidate_text = res.doc.text
            # Skip if Jaccard similarity is too high (likely is the same answer or a duplicate)
            similarity = jaccard_similarity(positive, candidate_text)
            if similarity > args.overlap_threshold:
                skipped_high_overlap += 1
                continue
            
            hard_negative = candidate_text
            break

        if hard_negative:
            triplets.append({
                "query": query,
                "positive_passage": positive,
                "negative_passage": hard_negative
            })
            pbar.update(1)
        else:
            skipped_no_negative += 1

    pbar.close()

    print(f"Mined {len(triplets)} triplets successfully.")
    print(f"Skipped (high overlap): {skipped_high_overlap}")
    print(f"Skipped (no negative found): {skipped_no_negative}")

    # Save as JSONL
    print(f"Saving triplets to {args.output}...")
    with args.output.open("w", encoding="utf-8") as f:
        import json
        for trip in triplets:
            f.write(json.dumps(trip, ensure_ascii=False) + "\n")
    print("Mined triplets saved successfully.")


if __name__ == "__main__":
    main()
