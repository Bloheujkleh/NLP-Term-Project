# Demo Talking Points

## One-Minute Summary

This project builds a Turkish legal RAG system that answers legal questions only from retrieved source passages and displays citations. The strongest live demo configuration is BM25 retrieval with an extractive grounded answer generator.

## What To Show

1. Open the browser demo.
2. Ask one of the prepared questions.
3. Point to the answer and its `Kaynak:` line.
4. Show the retrieved sources below the answer.
5. Explain that the answer is not free-form guessing; it is grounded in the displayed source.

## Safe Demo Questions

```text
Kasten oldurme sucu nedir?
Adil yargilanma hakki nasil guvence altina alinir?
Evlilik birligi temelinden sarsilirsa ne olur?
```

## Key Metrics

| Metric | Result |
|---|---:|
| BM25 Recall@10 | 0.975 |
| QA Token F1 | 0.799 |
| Top-5 Source Hit | 0.908 |
| Citation Accuracy | 0.813 |
| NLI Judge Faithfulness | 0.858 |

## If Asked About Fine-Tuning

- Embedding fine-tuning was run on CPU with all 2,059 triples.
- It did not improve dense retrieval, so BM25 stayed as the live demo retriever.
- Reranker fine-tuning was run on CPU with all 6,752 pairs.
- It improved strongly over the pretrained reranker, but still did not beat BM25 on the full benchmark.
- FLAN-T5 SFT smoke training was run with `llm.jsonl`.
- The small CPU model was not citation-reliable enough for the final demo.

## Best Closing Line

The main result is not just a demo, but a measured RAG pipeline: we compared retrieval methods, ran fine-tuning experiments, measured citation and faithfulness, analyzed errors, and selected the safest configuration for legal QA.
