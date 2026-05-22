# Step 6 - LLM/SFT Smoke Experiment

## Goal

The assignment includes an LLM fine-tuning component. Full Turkish legal LLM fine-tuning is GPU-heavy, so a CPU-safe smoke experiment was added to verify that `llm.jsonl` can be used for source-grounded instruction tuning.

## Implemented Script

Script:

```bash
scripts/train_seq2seq_generator.py
```

The script converts each `llm.jsonl` row from chat format into a seq2seq training pair:

- input: system instruction + user source/question prompt
- target: assistant source-grounded answer

It fine-tunes a local `google/flan-t5-small` seq2seq model with HuggingFace `Seq2SeqTrainer`.

## CPU Smoke Run

Command:

```bash
python scripts/train_seq2seq_generator.py \
  --limit 512 \
  --eval-size 64 \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-input-length 512 \
  --max-target-length 160 \
  --output-dir outputs/models/flan_t5_legal_sft_smoke_512 \
  --metrics-output outputs/llm_sft_smoke_512_metrics.json
```

Training summary:

| Setting | Value |
|---|---:|
| Training examples | 448 |
| Evaluation examples | 64 |
| Epochs | 1 |
| Runtime | 629 seconds |
| Training loss | 7.368 |
| Eval loss | 0.514 |

## Small QA Check

The smoke model was evaluated on 20 gold QA examples using BM25 retrieval and local HuggingFace generation:

```bash
python scripts/evaluate_qa.py \
  --retriever bm25 \
  --generation-mode local_hf \
  --generation-model outputs/models/flan_t5_legal_sft_smoke_512 \
  --top-k 3 \
  --limit 20 \
  --max-new-tokens 96 \
  --output outputs/qa_eval_local_hf_sft_smoke_20.json
```

| Metric | Value |
|---|---:|
| Exact match | 0.000 |
| Token F1 | 0.103 |
| ROUGE-L | 0.075 |
| Citation accuracy | 0.000 |
| Faithfulness proxy | 0.616 |

## Interpretation

The smoke experiment verifies the LLM fine-tuning pipeline, model saving, and local generation path. However, the small CPU-trained FLAN-T5 model is not strong enough for the final demo: it produces weak answers and does not reliably preserve citations.

Therefore, the final demo keeps the extractive source-grounded generator. For a production-quality generative answerer, the next step is GPU-based fine-tuning of a stronger Turkish-capable instruction model and stricter citation-format training.
