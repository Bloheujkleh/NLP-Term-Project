from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def read_summary(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Base RAG and Fine-tuned RAG on the same corpus, benchmark, and answer generator"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submission_eval"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generation-mode", choices=["extractive", "local_hf", "ollama"], default="extractive")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--base-embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--finetuned-retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--finetuned-embedding-model", default=None)
    parser.add_argument("--finetuned-reranker-model", default=None)
    parser.add_argument("--rerank-top-n", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_output = args.output_dir / "base_rag_qa.json"
    finetuned_output = args.output_dir / "finetuned_rag_qa.json"
    comparison_output = args.output_dir / "base_vs_finetuned_summary.json"

    common = [
        sys.executable,
        "scripts/evaluate_qa.py",
        "--data-dir",
        str(args.data_dir),
        "--top-k",
        str(args.top_k),
        "--generation-mode",
        args.generation_mode,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.generation_model:
        common += ["--generation-model", args.generation_model]
    if args.limit:
        common += ["--limit", str(args.limit)]

    base_cmd = common + [
        "--retriever",
        args.base_retriever,
        "--embedding-model",
        args.base_embedding_model,
        "--output",
        str(base_output),
    ]

    finetuned_cmd = common + [
        "--retriever",
        args.finetuned_retriever,
        "--output",
        str(finetuned_output),
    ]
    if args.finetuned_embedding_model:
        finetuned_cmd += ["--embedding-model", args.finetuned_embedding_model]
    else:
        finetuned_cmd += ["--embedding-model", args.base_embedding_model]
    if args.finetuned_reranker_model:
        finetuned_cmd += [
            "--reranker-model",
            args.finetuned_reranker_model,
            "--rerank-top-n",
            str(args.rerank_top_n),
        ]

    run_command(base_cmd)
    run_command(finetuned_cmd)

    base_summary = read_summary(base_output)
    finetuned_summary = read_summary(finetuned_output)
    deltas = {
        key: finetuned_summary.get(key, 0.0) - base_summary.get(key, 0.0)
        for key in sorted(set(base_summary) | set(finetuned_summary))
    }
    comparison = {
        "note": "Both systems were evaluated on the same corpus, benchmark, and answer generator.",
        "base_config": {
            "retriever": args.base_retriever,
            "embedding_model": args.base_embedding_model if args.base_retriever != "bm25" else None,
            "generation_mode": args.generation_mode,
            "generation_model": args.generation_model,
        },
        "finetuned_config": {
            "retriever": args.finetuned_retriever,
            "embedding_model": (args.finetuned_embedding_model or args.base_embedding_model)
            if args.finetuned_retriever != "bm25"
            else None,
            "reranker_model": args.finetuned_reranker_model,
            "generation_mode": args.generation_mode,
            "generation_model": args.generation_model,
        },
        "base_summary": base_summary,
        "finetuned_summary": finetuned_summary,
        "delta_finetuned_minus_base": deltas,
    }
    with comparison_output.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("\nComparison summary")
    for key in sorted(deltas):
        print(f"{key}: base={base_summary.get(key, 0.0):.4f} fine={finetuned_summary.get(key, 0.0):.4f} delta={deltas[key]:+.4f}")
    print(f"Wrote {comparison_output}")


if __name__ == "__main__":
    main()
