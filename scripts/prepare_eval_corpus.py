from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a smaller 1,000-doc evaluation corpus or restore the full corpus.")
    parser.add_argument("--action", choices=["shrink", "restore"], default="shrink")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    real_path = args.data_dir / "real_corpus.jsonl"
    real_full_path = args.data_dir / "real_corpus_full.jsonl"
    index_path = args.data_dir / "corpus_index.jsonl"
    index_full_path = args.data_dir / "corpus_index_full.jsonl"
    qa_path = args.data_dir / "eval_qa_150.jsonl"

    if args.action == "shrink":
        # 1. Back up full files if not already done
        if not real_full_path.exists() and real_path.exists():
            real_path.rename(real_full_path)
            print(f"Backed up real_corpus.jsonl to {real_full_path}")
        
        if not index_full_path.exists() and index_path.exists():
            index_path.rename(index_full_path)
            print(f"Backed up corpus_index.jsonl to {index_full_path}")
        elif index_path.exists():
            index_path.unlink()  # delete the active index file so it doesn't get loaded

        if not real_full_path.exists():
            print(f"Error: {real_full_path} does not exist. Cannot shrink.")
            return

        # 2. Load eval QA to find gold source IDs
        gold_ids = set()
        with qa_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "source_id" in item:
                        gold_ids.add(item["source_id"])
        print(f"Loaded {len(gold_ids)} gold source IDs from {qa_path}")

        # 3. Read full corpus
        gold_docs = []
        other_docs = []
        with real_full_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item["id"] in gold_ids:
                        gold_docs.append(item)
                    else:
                        other_docs.append(item)

        print(f"Found {len(gold_docs)} gold docs in full corpus. {len(other_docs)} other docs.")

        # 4. Sample distractors
        num_distractors = max(0, args.limit - len(gold_docs))
        sampled_others = random.sample(other_docs, min(num_distractors, len(other_docs)))
        
        eval_corpus = gold_docs + sampled_others
        # Shuffle to mix them up
        random.shuffle(eval_corpus)

        # 5. Save as the active real_corpus.jsonl
        with real_path.open("w", encoding="utf-8") as f:
            for item in eval_corpus:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"Created eval corpus with {len(eval_corpus)} documents in {real_path}")

    elif args.action == "restore":
        # Restore files
        if real_full_path.exists():
            if real_path.exists():
                real_path.unlink()
            real_full_path.rename(real_path)
            print(f"Restored real_corpus.jsonl from {real_full_path}")
        
        if index_full_path.exists():
            if index_path.exists():
                index_path.unlink()
            index_full_path.rename(index_path)
            print(f"Restored corpus_index.jsonl from {index_full_path}")

if __name__ == "__main__":
    main()
