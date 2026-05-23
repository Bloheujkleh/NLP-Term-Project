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


def build_llama3_prompt(question: str, results: list[SearchResult]) -> str:
    # Use top-3 results as context
    context_blocks = []
    for idx, res in enumerate(results[:3], start=1):
        context_blocks.append(
            f"[{idx}] Başlık: {res.doc.title}\nCitation: {res.doc.citation_label}\nMetin: {res.doc.text}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "Aşağıdaki hukuki bağlama dayanarak soruyu yanıtla. "
        "Eğer bağlamda soruya ait bir cevap yoksa 'Bu konuda bilgim yok' şeklinde yanıt ver.\n\n"
        f"Bağlam:\n{context}\n\n"
        f"Soru: {question}\n\n"
        "Cevap:"
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
    max_new_tokens: int = 256
    temperature: float = 0.0


class AnswerGenerator:
    def generate(self, question: str, results: list[SearchResult]) -> str:
        raise NotImplementedError


class ExtractiveAnswerGenerator(AnswerGenerator):
    def generate(self, question: str, results: list[SearchResult]) -> str:
        return extractive_baseline_answer(question, results)


class LocalHFAnswerGenerator(AnswerGenerator):
    def __init__(self, config: GenerationConfig) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

        if not config.model_name:
            raise ValueError("--generation-model is required for local_hf mode")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Determine if causal (Decoder-only like Llama) or seq2seq (Encoder-Decoder like T5)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
            self.is_causal = True
        except Exception:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
            self.is_causal = False

    def generate(self, question: str, results: list[SearchResult]) -> str:
        prompt = build_llama3_prompt(question, results)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-5),
        )
        if self.is_causal:
            # Decode only the generated part
            input_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[0][input_len:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


class OllamaAnswerGenerator(AnswerGenerator):
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.model_name = config.model_name or "llama3"
        self.api_url = "http://localhost:11434/api/generate"

    def generate(self, question: str, results: list[SearchResult]) -> str:
        import json
        import urllib.request
        import urllib.error

        prompt = build_llama3_prompt(question, results)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_new_tokens
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "").strip()
        except urllib.error.URLError as e:
            return f"Error: Ollama connection failed. Make sure Ollama is running. Error: {e}"
        except Exception as e:
            return f"Error: Unexpected error during Ollama generation: {e}"


def build_answer_generator(config: GenerationConfig) -> AnswerGenerator:
    if config.mode == "extractive":
        return ExtractiveAnswerGenerator()
    if config.mode == "local_hf":
        return LocalHFAnswerGenerator(config)
    if config.mode == "ollama":
        return OllamaAnswerGenerator(config)
    raise ValueError(f"Unknown generation mode: {config.mode}")
