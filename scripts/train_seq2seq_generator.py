from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from legal_rag.data import read_jsonl, write_json


def row_to_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    messages = row.get("messages") or []
    system = ""
    user = ""
    assistant = ""
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == "system":
            system = content
        elif role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    if not user or not assistant:
        return None
    prompt = f"{system}\n\n{user}\n\nCevap:" if system else f"{user}\n\nCevap:"
    return prompt, assistant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets_Ceng493_legal_rag"))
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/flan_t5_legal_sft_smoke"))
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/llm_sft_smoke_metrics.json"))
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.data_dir / "llm.jsonl")
    pairs = [pair for row in rows for pair in [row_to_pair(row)] if pair is not None]
    if args.limit:
        pairs = pairs[: args.limit]

    eval_size = min(args.eval_size, max(1, len(pairs) // 5))
    train_pairs = pairs[:-eval_size]
    eval_pairs = pairs[-eval_size:]

    train_ds = Dataset.from_dict(
        {
            "input_text": [pair[0] for pair in train_pairs],
            "target_text": [pair[1] for pair in train_pairs],
        }
    )
    eval_ds = Dataset.from_dict(
        {
            "input_text": [pair[0] for pair in eval_pairs],
            "target_text": [pair[1] for pair in eval_pairs],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(batch: dict[str, list[str]]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)

    output = {
        "config": {
            "model_name": args.model_name,
            "training_examples": len(train_pairs),
            "eval_examples": len(eval_pairs),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
        },
        "train": {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))},
        "eval": {key: float(value) for key, value in eval_result.items() if isinstance(value, (int, float))},
        "output_dir": str(args.output_dir),
    }
    write_json(args.metrics_output, output)
    print("Seq2seq SFT smoke training complete")
    print(output)


if __name__ == "__main__":
    main()
