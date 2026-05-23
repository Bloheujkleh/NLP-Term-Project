from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from legal_rag.data import read_jsonl
from legal_rag.retrievers import DEFAULT_EMBEDDING_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--base-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/legal_embedding_triplet"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(args.data_dir / "embedding.jsonl")
    random.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    is_e5 = "e5" in args.base_model.lower()
    examples = []
    for row in rows:
        q = row["query"]
        pos = row["positive_passage"]
        neg = row["negative_passage"]
        if is_e5:
            q = f"query: {q}"
            pos = f"passage: {pos}"
            neg = f"passage: {neg}"
        examples.append(InputExample(texts=[q, pos, neg]))

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = args.warmup_steps

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned embedding model to {args.output_dir}")


if __name__ == "__main__":
    main()

