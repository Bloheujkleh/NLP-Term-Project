from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import read_json, write_json
from legal_rag.metrics import extract_answer_body, lexical_faithfulness_proxy, token_f1


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def parse_yes_no(text: str) -> float:
    normalized = text.strip().lower()
    yes_markers = ["yes", "supported", "faithful", "true", "1", "evet"]
    no_markers = ["no", "unsupported", "not faithful", "false", "0", "hayir", "hayır"]
    if any(marker in normalized for marker in no_markers):
        return 0.0
    if any(marker in normalized for marker in yes_markers):
        return 1.0
    return 0.5


def build_prompt(row: dict[str, Any], max_context_chars: int) -> str:
    return (
        "Turkish legal QA evaluation.\n"
        "Task: Is the answer supported by the source text? Reply only YES or NO.\n\n"
        f"Source text: {compact(row.get('context', ''), max_context_chars)}\n"
        f"Question: {compact(row['question'], 350)}\n"
        f"Answer: {compact(row['answer'], 700)}\n"
        "Supported by source? YES or NO:"
    )


def first_source_text(context: str) -> str:
    if "[2] Başlık:" in context:
        context = context.split("[2] Başlık:", 1)[0]
    marker = "Metin:"
    if marker in context:
        context = context.split(marker, 1)[1]
    return context.strip()


class LocalFlanJudge:
    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def judge(self, prompt: str) -> tuple[float, str]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return parse_yes_no(text), text


class OpenAIJudge:
    def __init__(self, model_name: str, max_retries: int = 3) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries

    def judge(self, prompt: str) -> tuple[float, str]:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a strict legal QA evaluator. Reply only YES or NO."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 4,
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"]
                return parse_yes_no(text), text
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI judge failed: {last_error}")


class NLIJudge:
    def __init__(self, model_name: str, max_length: int, entailment_threshold: float) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.max_length = max_length
        self.entailment_threshold = entailment_threshold
        self.id2label = {int(key): value.lower() for key, value in self.model.config.id2label.items()}

    def judge_row(self, row: dict[str, Any], max_context_chars: int) -> tuple[float, str]:
        import torch

        premise = compact(first_source_text(row.get("context", "")), max_context_chars)
        hypothesis = compact(extract_answer_body(row["answer"]), 700)
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).tolist()
        scores = {self.id2label[i]: probabilities[i] for i in range(len(probabilities))}
        entailment = scores.get("entailment", 0.0)
        label = float(entailment >= self.entailment_threshold)
        raw = ", ".join(f"{name}={value:.3f}" for name, value in sorted(scores.items()))
        return label, raw


def heuristic_judge(row: dict[str, Any]) -> tuple[float, str]:
    faithfulness = lexical_faithfulness_proxy(row["answer"], [row.get("context", "")])
    answer_f1 = token_f1(row["answer"], row["reference"])
    label = float(faithfulness >= 0.75 and answer_f1 >= 0.35)
    return label, f"heuristic faithfulness={faithfulness:.3f}, answer_f1={answer_f1:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/qa_eval_extractive_bm25_full.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/llm_judge_faithfulness.json"))
    parser.add_argument("--provider", choices=["local-flan", "openai", "nli", "heuristic"], default="nli")
    parser.add_argument("--model", default="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-chars", type=int, default=900)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--nli-max-length", type=int, default=512)
    parser.add_argument("--entailment-threshold", type=float, default=0.5)
    args = parser.parse_args()

    qa = read_json(args.input)
    rows = qa["answers"]
    if args.limit:
        rows = rows[: args.limit]

    if args.provider == "local-flan":
        judge = LocalFlanJudge(args.model, args.max_new_tokens)
        judge_fn = lambda row: judge.judge(build_prompt(row, args.max_context_chars))
    elif args.provider == "openai":
        judge = OpenAIJudge(args.model)
        judge_fn = lambda row: judge.judge(build_prompt(row, args.max_context_chars))
    elif args.provider == "nli":
        judge = NLIJudge(args.model, args.nli_max_length, args.entailment_threshold)
        judge_fn = lambda row: judge.judge_row(row, args.max_context_chars)
    else:
        judge_fn = heuristic_judge

    evaluated = []
    total = 0.0
    yes_count = 0
    no_count = 0
    uncertain_count = 0

    for index, row in enumerate(rows, start=1):
        score, raw = judge_fn(row)
        total += score
        yes_count += int(score == 1.0)
        no_count += int(score == 0.0)
        uncertain_count += int(score == 0.5)
        evaluated.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "answer": row["answer"],
                "reference": row["reference"],
                "judge_score": score,
                "judge_raw": raw,
                "token_f1": row["metrics"]["token_f1"],
                "citation_label_accuracy": row["metrics"]["citation_label_accuracy"],
                "faithfulness_proxy": row["metrics"]["faithfulness_proxy"],
            }
        )
        if index % 25 == 0:
            print(f"Judged {index}/{len(rows)}")

    output = {
        "config": {
            "provider": args.provider,
            "model": args.model,
            "num_questions": len(rows),
            "input": str(args.input),
        },
        "summary": {
            "judge_faithfulness": total / len(rows) if rows else 0.0,
            "judge_yes": yes_count,
            "judge_no": no_count,
            "judge_uncertain": uncertain_count,
        },
        "judgments": evaluated,
    }
    write_json(args.output, output)
    print("LLM judge evaluation complete")
    print(output["config"])
    print(output["summary"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
