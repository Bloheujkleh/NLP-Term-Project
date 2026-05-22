# Code Evidence Map

This file maps the main project claims to the code that implements them. It is intended for quick instructor/demo defense.

## Retrieval

| Claim | Code |
|---|---|
| BM25 retrieval is implemented and evaluated. | `src/legal_rag/retrievers.py`, `scripts/evaluate_retrieval.py` |
| Dense vector retrieval uses sentence embeddings and FAISS. | `src/legal_rag/retrievers.py`, `scripts/evaluate_retrieval.py` |
| Hybrid retrieval combines BM25 and dense scores. | `src/legal_rag/retrievers.py`, `scripts/evaluate_retrieval.py` |

## Fine-Tuning

| Claim | Code |
|---|---|
| Embedding fine-tuning uses query, positive passage, negative passage triples. | `scripts/train_embedding_model.py` |
| Reranker fine-tuning uses query, candidate passage, binary label pairs. | `scripts/train_cross_encoder_reranker.py` |
| FLAN-T5 SFT smoke uses source-grounded chat-style examples from `llm.jsonl`. | `scripts/train_seq2seq_generator.py` |

## Evaluation

| Claim | Code |
|---|---|
| Retrieval metrics include Recall@5, Recall@10, MRR, and nDCG@10. | `src/legal_rag/metrics.py`, `scripts/evaluate_retrieval.py` |
| QA metrics include EM, token F1, ROUGE-L, source hit, citation accuracy, and lexical faithfulness proxy. | `src/legal_rag/metrics.py`, `scripts/evaluate_qa.py` |
| Reranker evaluation keeps BM25 as the first-stage retriever and changes only the cross-encoder. | `scripts/evaluate_reranker.py` |
| NLI-based semantic faithfulness judge is implemented separately from the lexical proxy. | `scripts/evaluate_llm_judge.py` |

## Demo

| Claim | Code |
|---|---|
| Browser demo is a working source-grounded QA system. | `scripts/demo_app.py` |
| Browser demo can run either extractive answers or the local fine-tuned FLAN-T5 generator. | `scripts/demo_app.py` |
| CLI fallback returns answer and sources without a browser. | `scripts/demo_cli.py` |
| CLI fallback can run the same optional local FLAN-T5 generator path. | `scripts/demo_cli.py` |
| Fast smoke test checks corpus loading, BM25 retrieval, citation output, and metric sanity. | `scripts/smoke_test.py` |

## Important Limitation

The default live demo intentionally uses the reliable extractive grounded generator. The generative FLAN-T5 path is also connected through `--answer-mode local_hf` and was fine-tuned/evaluated as a smoke experiment, but it is not the recommended demo path because citation accuracy remained weak.
