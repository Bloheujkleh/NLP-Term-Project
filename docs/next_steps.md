# Remaining Next Steps

Most implementation and experiment work for the course submission has been completed. This file only lists optional future improvements beyond the current demo-ready project.

## Current Completed Status

| Area | Status |
|---|---|
| Baseline RAG | Complete |
| BM25, dense, hybrid retrieval | Complete |
| Retrieval metrics | Complete |
| QA metrics and citation accuracy | Complete |
| Error analysis | Complete |
| Embedding fine-tuning | Full CPU run completed |
| Reranker fine-tuning | Full CPU run completed |
| Judge faithfulness | Full 240-example run completed |
| LLM/SFT training path | 512-example CPU smoke run completed |
| Browser demo | Complete |
| Report and deliverables | Complete |

## Optional Future Work

### Stronger Generative LLM

The current live demo uses extractive source-grounded answers because it is reliable and citation-safe. A future version can fine-tune a stronger Turkish-capable instruction model on `llm.jsonl` using GPU hardware.

Recommended evaluation:

- QA token F1 / ROUGE-L
- Citation accuracy
- NLI or LLM judge faithfulness
- Manual hallucination review

### Better Dense Retrieval

The CPU triplet-tuned dense retriever did not improve over the base multilingual MiniLM model. Future work should try:

- stronger Turkish or multilingual legal embedding models
- validation split for early stopping
- improved hard-negative mining
- longer sequence length on GPU
- multiple loss functions beyond the current simple triplet setup

### Reranker Deployment

The fine-tuned reranker improved strongly over the pretrained reranker, but did not beat direct BM25 on the full benchmark. Future work can test:

- different candidate counts
- longer max sequence length
- calibration with BM25 score interpolation
- reranker only on queries where BM25 confidence is low

## Final Demo Recommendation

For submission/demo, use the current BM25 + extractive grounded answer system. It is the most reliable measured configuration and gives auditable citations in the browser UI.
