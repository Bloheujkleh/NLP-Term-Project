"""Colab script: fine-tune Qwen2.5 for Turkish legal source-grounded RAG.

Run this in Google Colab with a GPU runtime. It trains a LoRA adapter on
`data/llm.jsonl`, merges the adapter into the base model, and optionally pushes
the merged model to Hugging Face Hub.

Recommended Colab runtime: T4 GPU or better.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TRAIN_FILE = Path(os.environ.get("TRAIN_FILE", "data/llm.jsonl"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs/qwen2_5_0_5b_legal_rag_lora"))
MERGED_DIR = Path(os.environ.get("MERGED_DIR", "outputs/qwen2_5_0_5b_legal_rag_merged"))
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")  # example: "bulent/turkish-legal-qwen-rag-sft"
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "12000"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1536"))


def load_messages_dataset(path: Path, tokenizer: AutoTokenizer, max_examples: int) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not messages:
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            rows.append({"text": text})
            if len(rows) >= max_examples:
                break
    return Dataset.from_list(rows)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])
    return tokenized


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_messages_dataset(TRAIN_FILE, tokenizer, MAX_EXAMPLES)
    split = dataset.train_test_split(test_size=0.03, seed=42)
    train_ds = tokenize_dataset(split["train"], tokenizer)
    eval_ds = tokenize_dataset(split["test"], tokenizer)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Merge LoRA into a normal model repo so deployment does not need PEFT.
    del model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    merged = merged.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    if HF_REPO_ID:
        merged.push_to_hub(HF_REPO_ID, safe_serialization=True)
        tokenizer.push_to_hub(HF_REPO_ID)
        print(f"\nPushed merged fine-tuned model to: {HF_REPO_ID}")
        print(f"Set Space env GENERATION_MODEL={HF_REPO_ID}")
    else:
        print(f"\nMerged model saved locally at: {MERGED_DIR}")
        print("Set HF_REPO_ID to push it to Hugging Face Hub.")


if __name__ == "__main__":
    main()
