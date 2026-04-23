"""
Local answer generator for RAG using HuggingFace Transformers (no pipeline).
Uses AutoTokenizer + AutoModelForSeq2SeqLM + model.generate for Flan-T5.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Required instruction for constrained legal QA.
PROMPT_INSTRUCTION = (
    "Answer the question using only the provided legal context. "
    "If the answer is not contained in the context, say that the information is insufficient."
)


def _concatenate_contexts(contexts: List[str]) -> str:
    """Join retrieved chunks with clear separators; skip empty strings."""
    parts = [c.strip() for c in contexts if c and str(c).strip()]
    if not parts:
        return "(no context provided)"
    return "\n\n---\n\n".join(parts)


def _extractive_fallback(question: str, contexts: List[str]) -> str:
    """If model load or generation fails, return a short snippet from the top context."""
    if not contexts:
        return "The information is insufficient based on the retrieved context."

    top = (contexts[0] or "").strip()
    if not top:
        return "The information is insufficient based on the retrieved context."

    parts = re.split(r"(?<=[.!?])\s+", top, maxsplit=1)
    first = parts[0].strip() if parts else top

    max_chars = 500
    if len(first) > max_chars:
        cut = first[:max_chars].rsplit(" ", 1)[0]
        return f"{cut}…"

    if len(first) < 80 and len(top) > len(first):
        snippet = top[:max_chars].rsplit(" ", 1)[0]
        return f"{snippet}…" if len(snippet) < len(top) else snippet

    return first


def _build_prompt(question: str, contexts: List[str]) -> str:
    """
    Single prompt string: instruction + question + merged legal context.
    """
    context_block = _concatenate_contexts(contexts)
    q = question.strip()
    return (
        f"{PROMPT_INSTRUCTION}\n\n"
        f"Question:\n{q}\n\n"
        f"Legal Context:\n{context_block}\n\n"
        f"Answer:"
    )


class LocalGenerator:
    """
    Seq2seq generation for Flan-T5 style models (encoder-decoder).
    Default: google/flan-t5-small
    """

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            path = Path(model_name)
            adapter_cfg = path / "adapter_config.json"
            if adapter_cfg.exists():
                cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                base = cfg.get("base_model_name_or_path", "google/flan-t5-small")
                self._tokenizer = AutoTokenizer.from_pretrained(base)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base)
                self._model = PeftModel.from_pretrained(base_model, str(path))
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()
        except Exception:
            self._tokenizer = None
            self._model = None

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        Generate an answer from the question and retrieved context strings.
        On load/generate failure, returns an extractive fallback from the first context.
        """
        if self._tokenizer is None or self._model is None:
            return _extractive_fallback(question, contexts)

        prompt = _build_prompt(question, contexts)

        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=128,
                    do_sample=False,
                )

            text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text = (text or "").strip()
            if not text:
                return _extractive_fallback(question, contexts)
            return text
        except Exception:
            return _extractive_fallback(question, contexts)
