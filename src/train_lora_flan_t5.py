"""
LoRA fine-tuning for FLAN-T5 (seq2seq) on Turkish legal QA.

Task format (instructional):
  input: prompt with legal context + question
  target: gold answer (Cevap)

Notes:
- Intended for GPU training. CPU will be very slow.
- Uses PEFT LoRA on the encoder+decoder attention projections.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from generator import PROMPT_INSTRUCTION, _concatenate_contexts


def build_prompt(question: str, context: str) -> str:
    return (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Legal Context:\n{context.strip()}\n\n"
        f"Answer:"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--out_dir", default="models/flan-t5-small-lora-legal")
    parser.add_argument("--max_train", type=int, default=4000)
    parser.add_argument("--max_eval", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = load_dataset("Renicames/turkish-law-chatbot")
    train_hf = ds["train"]
    eval_hf = ds["test"]

    train_rows = train_hf.select(range(min(args.max_train, len(train_hf))))
    eval_rows = eval_hf.select(range(min(args.max_eval, len(eval_hf))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v"],
    )
    model = get_peft_model(model, lora_config)

    def rows_to_dataset(split) -> Dataset:
        inputs: List[str] = []
        targets: List[str] = []
        for row in split:
            q = str(row["Soru"]).strip()
            a = str(row["Cevap"]).strip()
            if not q or not a:
                continue
            # Training proxy context: use the gold answer text as a compact "legal context" snippet.
            # This is not identical to retrieval-augmented inference, but provides a stable SFT signal.
            ctx = a
            inputs.append(build_prompt(q, _concatenate_contexts([ctx])))
            targets.append(a)
        return Dataset.from_dict({"input_text": inputs, "target_text": targets})

    train_ds = rows_to_dataset(train_rows)
    eval_ds = rows_to_dataset(eval_rows)

    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(Path(args.out_dir) / "trainer_out"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=50,
        evaluation_strategy="no",
        save_strategy="no",
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[LoRA] Saved adapter+tokenizer -> {out}")


if __name__ == "__main__":
    main()
