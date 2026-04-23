"""
Contrastive fine-tuning for dense retrieval (sentence-transformers).

Pairs:
  (query=Soru, passage=Cevap) from HF ``train`` split

Hard negatives (simple):
  For each query, take one random other row's Cevap as a negative passage.

Output:
  A local folder suitable for SentenceTransformer loading, e.g. models/st-legal-v1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def build_examples(max_samples: int, seed: int) -> List[InputExample]:
    random.seed(seed)
    ds = load_dataset("Renicames/turkish-law-chatbot")
    train = ds["train"]

    texts_q: List[str] = []
    texts_p: List[str] = []
    for row in train:
        q = str(row.get("Soru", "")).strip()
        p = str(row.get("Cevap", "")).strip()
        if not q or not p:
            continue
        texts_q.append(q)
        texts_p.append(p)

    n = len(texts_q)
    if n == 0:
        raise RuntimeError("No usable HF train rows.")

    limit = min(max_samples, n)
    idxs = list(range(n))
    random.shuffle(idxs)
    idxs = idxs[:limit]

    examples: List[InputExample] = []
    for i in idxs:
        q = texts_q[i]
        pos = texts_p[i]
        neg_j = random.randint(0, n - 1)
        if neg_j == i:
            neg_j = (neg_j + 1) % n
        neg = texts_p[neg_j]
        examples.append(InputExample(texts=[q, pos, neg]))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--out_dir", default="models/st-legal-multilingual-v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=8000)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    examples = build_examples(args.max_samples, args.seed)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    model = SentenceTransformer(args.base_model)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=int(len(train_dataloader) * args.warmup_ratio),
        show_progress_bar=True,
        output_path=str(out),
    )

    print(f"[EmbeddingFT] Saved model -> {out}")


if __name__ == "__main__":
    main()
