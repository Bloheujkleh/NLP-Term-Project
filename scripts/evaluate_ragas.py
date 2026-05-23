from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.data import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers using RAGAS or local fallback")
    parser.add_argument("--input", type=Path, default=Path("outputs/qa_eval.json"), 
                        help="JSON output file from evaluate_qa.py")
    parser.add_argument("--output", type=Path, default=Path("outputs/ragas_eval_results.json"))
    parser.add_argument("--provider", choices=["openai", "ollama", "local-nli"], default="local-nli",
                        help="LLM provider for RAGAS evaluation (openai, ollama, or local NLI fallback)")
    parser.add_argument("--model", default="llama3", help="Ollama model name if provider is ollama")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist. Please run evaluate_qa.py first.")
        sys.exit(1)

    print(f"Loading QA evaluations from {args.input}...")
    qa_data = read_json(args.input)
    answers = qa_data.get("answers", [])
    if args.limit:
        answers = answers[:args.limit]

    if not answers:
        print("Error: No answers found in the input file.")
        sys.exit(1)

    # Prepare dataset for evaluation
    questions = [item["question"] for item in answers]
    generated_answers = [item["answer"] for item in answers]
    references = [item["reference"] for item in answers]
    
    # Retrieve contexts (Ragas expects a list of lists of strings)
    # The context in input json is stored as a compiled string 'build_context', let's reconstruct lists
    contexts = []
    for item in answers:
        # Extract list of texts from build_context string or retrieved_ids
        context_str = item.get("context", "")
        # Split by '[i] Başlık:' to separate chunks
        chunks = []
        if "[1] Başlık:" in context_str:
            parts = context_str.split("[")
            for part in parts[1:]:
                # Extract text after 'Metin: '
                if "Metin:" in part:
                    text_part = part.split("Metin:", 1)[1].strip()
                    chunks.append(text_part)
        else:
            chunks = [context_str]
        contexts.append(chunks)

    print(f"Loaded {len(answers)} records. Starting evaluation using provider '{args.provider}'...")

    faithfulness_scores = []
    relevance_scores = []

    # 1. RAGAS using OpenAI or Ollama
    if args.provider in ["openai", "ollama"] and os.environ.get("OPENAI_API_KEY") or args.provider == "ollama":
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevance

            # Build Dataset
            eval_dataset = Dataset.from_dict({
                "question": questions,
                "answer": generated_answers,
                "contexts": contexts,
                "ground_truth": references
            })

            # Configure LLM
            if args.provider == "openai":
                print("Configuring OpenAI RAGAS evaluator...")
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            else:
                print(f"Configuring Ollama RAGAS evaluator with model '{args.model}'...")
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(model=args.model, temperature=0, base_url="http://localhost:11434")

            # Run RAGAS
            print("Running RAGAS evaluation...")
            result = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, answer_relevance],
                llm=llm
            )
            print("RAGAS Evaluation completed successfully.")
            print(result)

            # Store scores
            for i, item in enumerate(answers):
                faithfulness_scores.append(result["faithfulness"][i] if "faithfulness" in result else 0.0)
                relevance_scores.append(result["answer_relevance"][i] if "answer_relevance" in result else 0.0)

        except Exception as e:
            print(f"RAGAS execution failed: {e}. Falling back to Local NLI evaluator...")
            args.provider = "local-nli"

    # 2. Local NLI Fallback (Heuristic Faithfulness Judge)
    if args.provider == "local-nli" or not faithfulness_scores:
        print("Using Local Multilingual NLI Judge for Faithfulness...")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

            for idx, (question, answer, ctx_list) in enumerate(zip(questions, generated_answers, contexts)):
                premise = " ".join(ctx_list)[:900]
                # Strip citation prefixes from answer if present
                clean_answer = answer.split("\n\nKaynak:", 1)[0].strip()
                hypothesis = clean_answer[:500]

                inputs = tokenizer(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    logits = model(**inputs).logits[0]
                    probs = torch.softmax(logits, dim=-1).tolist()
                
                scores = {id2label[i]: probs[i] for i in range(len(probs))}
                entailment = scores.get("entailment", 0.0)
                faithfulness_scores.append(entailment)

                # Heuristic Answer Relevance (Jaccard similarity between generated answer and question)
                # This is a proxy for answer relevance in local environments
                q_words = set(question.lower().split())
                a_words = set(clean_answer.lower().split())
                overlap = len(q_words & a_words) / len(q_words | a_words) if q_words else 0.0
                relevance_scores.append(overlap)

                if (idx + 1) % 25 == 0:
                    print(f"Evaluated {idx + 1}/{len(answers)} records.")

        except Exception as e:
            print(f"Local NLI evaluation failed: {e}")
            # Heuristic token overlap fallback
            print("Falling back to simple Jaccard overlap faithfulness proxy...")
            for question, answer, ctx_list in zip(questions, generated_answers, contexts):
                clean_answer = answer.split("\n\nKaynak:", 1)[0].strip()
                a_words = set(clean_answer.lower().split())
                ctx_words = set(" ".join(ctx_list).lower().split())
                overlap = len(a_words & ctx_words) / len(a_words) if a_words else 0.0
                faithfulness_scores.append(overlap)
                relevance_scores.append(0.5)

    # Save results
    updated_answers = []
    total_faithfulness = 0.0
    total_relevance = 0.0

    for item, f_score, r_score in zip(answers, faithfulness_scores, relevance_scores):
        item["metrics"]["faithfulness"] = f_score
        item["metrics"]["answer_relevance"] = r_score
        total_faithfulness += f_score
        total_relevance += r_score
        updated_answers.append(item)

    n = len(answers)
    summary = qa_data.get("summary", {})
    summary["faithfulness"] = total_faithfulness / n
    summary["answer_relevance"] = total_relevance / n

    output_data = {
        "config": qa_data.get("config", {}),
        "summary": summary,
        "answers": updated_answers
    }
    
    # Copy configuration info
    output_data["config"]["evaluator_provider"] = args.provider

    write_json(args.output, output_data)
    print("\nEvaluation Summary:")
    for metric, val in summary.items():
        print(f"  {metric}: {val:.4f}")
    print(f"Wrote evaluation results to {args.output}")


if __name__ == "__main__":
    main()
