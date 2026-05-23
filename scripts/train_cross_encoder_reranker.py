from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

from legal_rag.data import read_jsonl
from legal_rag.rerankers import DEFAULT_RERANKER_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--base-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/legal_cross_encoder_reranker"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.data_dir / "reranker.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    examples = [
        InputExample(texts=[row["query"], row["candidate_passage"]], label=float(row["label"]))
        for row in rows
    ]
    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = math.ceil(len(train_loader) * args.epochs * args.warmup_ratio)

    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
    )
    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )
    if args.output_dir.exists() and not any(args.output_dir.iterdir()):
        shutil.rmtree(args.output_dir)
    model.save(str(args.output_dir))
    print(f"Saved fine-tuned reranker to {args.output_dir}")


if __name__ == "__main__":
    main()
