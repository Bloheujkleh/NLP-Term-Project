"""
Fine-tune a cross-encoder reranker for Turkish legal QA.

Training data (weak supervision):
- Positive: (Soru, Cevap) from HF train- Negative: (Soru, Cevap') where Cevap' is a random other answer

Output:
  A folder loadable by sentence_transformers.CrossEncoder
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader


def build_pairs(max_samples: int, seed: int) -> Tuple[List[InputExample], List[InputExample]]:
    random.seed(seed)
    ds = load_dataset("Renicames/turkish-law-chatbot")
    train = ds["train"]

    qs: List[str] = []
    ps: List[str] = []
    for row in train:
        q = str(row.get("Soru", "")).strip()
        p = str(row.get("Cevap", "")).strip()
        if not q or not p:
            continue
        qs.append(q)
        ps.append(p)

    n = len(qs)
    if n < 2:
        raise RuntimeError("Not enough HF train rows.")

    limit = min(max_samples, n)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[:limit]

    train_examples: List[InputExample] = []
    eval_examples: List[InputExample] = []

    for i, idx in enumerate(indices):
        q = qs[idx]
        pos = ps[idx]
        neg_j = random.randint(0, n - 1)
        if neg_j == idx:
            neg_j = (neg_j + 1) % n
        neg = ps[neg_j]

        ex_pos = InputExample(texts=[q, pos], label=1.0)
        ex_neg = InputExample(texts=[q, neg], label=0.0)

        # simple 90/10 split by modulo
        if i % 10 == 0:
            eval_examples.extend([ex_pos, ex_neg])
        else:
            train_examples.extend([ex_pos, ex_neg])

    return train_examples, eval_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--out_dir", default="models/ce-legal-v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ex, eval_ex = build_pairs(args.max_samples, args.seed)
    train_loader = DataLoader(train_ex, shuffle=True, batch_size=args.batch_size)

    model = CrossEncoder(args.base_model, num_labels=1)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(eval_ex, name="legal-dev")

    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=max(100, int(len(train_loader) * 0.1)),
        output_path=str(out),
        show_progress_bar=True,
    )

    print(f"[RerankerFT] Saved cross-encoder -> {out}")


if __name__ == "__main__":
    main()
