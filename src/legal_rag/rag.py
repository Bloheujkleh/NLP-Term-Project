from __future__ import annotations

from dataclasses import dataclass

from .retrievers import SearchResult


SYSTEM_PROMPT = (
    "Sen bir Türk hukuku RAG asistanısın. Yalnızca verilen kaynaklara dayanarak cevap ver. "
    "Kaynakta olmayan bilgiyi üretme ve cevabın sonunda kaynak belirt."
)


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        doc = result.doc
        blocks.append(
            f"[{i}] Başlık: {doc.title}\n"
            f"Chunk ID: {doc.id}\n"
            f"Citation: {doc.citation_label}\n"
            f"Metin: {doc.text}"
        )
    return "\n\n".join(blocks)


def build_grounded_prompt(question: str, results: list[SearchResult]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"[Kaynaklar]\n{build_context(results)}\n\n"
        f"[Soru]\n{question}\n\n"
        "Cevabı yalnızca kaynaklara dayanarak Türkçe ver. "
        "Cevabın sonunda kullandığın citation bilgisini 'Kaynak:' satırıyla yaz."
    )


def extractive_baseline_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "Bu soru için kaynak bulunamadı."
    best = results[0].doc
    return (
        f"Kaynağa göre: {best.text}\n\n"
        f"Kaynak: {best.citation_label}"
    )


@dataclass(frozen=True)
class GenerationConfig:
    mode: str = "extractive"
    model_name: str | None = None
    max_new_tokens: int = 192
    temperature: float = 0.0


class AnswerGenerator:
    def generate(self, question: str, results: list[SearchResult]) -> str:
        raise NotImplementedError


class ExtractiveAnswerGenerator(AnswerGenerator):
    def generate(self, question: str, results: list[SearchResult]) -> str:
        return extractive_baseline_answer(question, results)


class LocalHFAnswerGenerator(AnswerGenerator):
    def __init__(self, config: GenerationConfig) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if not config.model_name:
            raise ValueError("--generation-model is required for local_hf mode")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    def generate(self, question: str, results: list[SearchResult]) -> str:
        prompt = build_grounded_prompt(question, results)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-5),
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def build_answer_generator(config: GenerationConfig) -> AnswerGenerator:
    if config.mode == "extractive":
        return ExtractiveAnswerGenerator()
    if config.mode == "local_hf":
        return LocalHFAnswerGenerator(config)
    raise ValueError(f"Unknown generation mode: {config.mode}")
