# Turkish Legal RAG: Base vs Fine-Tuned RAG Evaluation

## 1. Objective

This project implements and evaluates a Retrieval-Augmented Generation (RAG) system for Turkish legal question answering. The goal is not only to produce fluent answers, but to produce answers that are grounded in legal sources, cite the supporting document, and can be evaluated on a gold benchmark.

The project is motivated by a practical limitation of standalone language models in legal domains. A general-purpose language model may answer confidently even when the answer is not supported by the relevant statute, legal article, or case text. In legal question answering, this is not only a quality problem but also a reliability and accountability problem. A correct-looking answer without a source is difficult to verify, and an unsupported answer can mislead the user.

RAG addresses this problem by separating the task into two stages. First, the system retrieves relevant legal documents from a controlled document collection. Second, the answer generator uses the retrieved documents as grounding context. This makes the answer auditable: the user can inspect which legal passage was used and whether the final answer is supported by that passage.

The instructor submission note defines four important requirements:

1. Compare Base RAG and Fine-tuned RAG systems using the same benchmark and, where applicable, the same LLM.
2. Use meaningful metrics depending on whether a gold benchmark is available.
3. Show ablation results for fine-tuned embedding, reranker, and LLM components where possible.
4. Support instructor-provided custom document collections and custom benchmark files.

This report is organized around those requirements.

The final system is therefore evaluated from three perspectives:

- Retrieval quality: whether the system finds the correct legal document.
- Answer quality: whether the produced answer matches the verified answer.
- Grounding quality: whether the answer is supported by and cites the retrieved source.

This framing is important because a RAG system can fail in different ways. It may retrieve the wrong document, retrieve the right document but rank it too low, produce an answer that does not match the gold answer, or cite an incorrect source. The evaluation in this report is designed to distinguish these failure modes.

## 2. Dataset and Evaluation Scenario

The project uses Turkish legal QA data with source documents, questions, verified answers, and relevant document identifiers. This matches Scenario 1 in the project rubric:

```text
Gold Question + Gold Answer + Gold Document
```

Because gold relevant documents exist, retrieval must be evaluated. Because gold answers exist, answer quality can be evaluated with lexical answer metrics. Because the task is legal QA, grounding and citation quality are also evaluated.

Main assignment dataset files used in the original benchmark:

| File | Size | Purpose |
|---|---:|---|
| `corpus.jsonl` | 7,579 rows | Legal source chunks used for retrieval |
| `rag_eval.json` | 1,000 rows | Retrieval benchmark with gold chunk IDs |
| `gold_benchmark.json` | 240 rows | QA benchmark with verified answers and gold sources |
| `embedding.jsonl` | 2,059 rows | Query-positive-negative triples for embedding tuning |
| `reranker.jsonl` | 6,752 rows | Query-passage-label pairs for reranker tuning |
| `llm.jsonl` | 13,758 rows | Source-grounded instruction tuning examples |

The latest repository also includes a custom-style local evaluation set:

| File | Size | Purpose |
|---|---:|---|
| `data/real_corpus.jsonl` | 15,094 rows | Larger local legal corpus |
| `data/eval_qa_150.jsonl` | 150 rows | Local custom-style QA benchmark |
| `sample_custom_data/corpus.jsonl` | 3 rows | Minimal custom corpus example |
| `sample_custom_data/eval_qa.jsonl` | 3 rows | Minimal custom benchmark example |

The main point is that the system can operate on a folder containing a custom `corpus.jsonl` and a custom benchmark file, not only on the original dataset.

Each corpus row represents a searchable legal passage. The passage-level representation is important because long legal documents are too large and too broad to retrieve as a single unit. If an entire law or long court decision is treated as one document, the retriever may return a source that is technically relevant but too large for precise answer grounding. By using chunks or passages, the retriever can return a more focused source.

The benchmark rows connect questions to both verified answers and relevant source documents. This allows the project to measure not only whether the final answer text is similar to the expected answer, but also whether the system actually found the correct supporting source. This is why the project uses source-hit and citation metrics in addition to answer-similarity metrics.

The distinction between training, evaluation, and demo data is also important. Fine-tuning data is used to adapt a model. Benchmark data is used to measure performance. Demo data is used to show the pipeline interactively. The same data should not be used to both train and claim unbiased evaluation for a component. In this project, the reported results are presented as controlled local experiments, and the final submission also supports instructor-provided benchmarks so that the system can be tested externally.

## 3. System Architecture

The RAG pipeline is:

```text
Question
-> Retriever
-> Top-k legal source chunks
-> Optional reranker
-> Answer generator
-> Answer with citation
```

Implemented retrieval components:

- BM25 lexical retrieval
- Dense vector retrieval with sentence-transformer embeddings
- Hybrid retrieval combining lexical and dense retrieval
- RRF-style rank fusion in the latest retrieval implementation

Implemented generation components:

- Extractive source-grounded answer generator
- Optional local HuggingFace generator using FLAN-T5
- Optional Ollama generation path

The default live demo uses:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

This is the most reliable measured live configuration. The Base vs Fine-tuned experiments below are still reported separately, as required by the submission note.

The architecture is intentionally modular. The retriever can be changed without changing the answer generator. The answer generator can be changed without changing the corpus format. The reranker can be inserted after first-stage retrieval. This modular design is what makes ablation possible: one component can be changed while the rest of the pipeline stays fixed.

The Base RAG and Fine-tuned RAG variants are therefore not separate applications. They are different configurations of the same pipeline. For example, one configuration may use BM25 directly, while another configuration may use BM25 candidates followed by a fine-tuned reranker. Another configuration may use the same BM25 retrieval but compare a base FLAN-T5 generator with a fine-tuned FLAN-T5 generator.

## 4. Metrics

Because this project has gold questions, gold answers, and gold documents, the metrics are selected according to Scenario 1 in the rubric.

Retrieval metrics:

| Metric | Purpose |
|---|---|
| Recall@5 / Recall@10 | Measures whether the relevant document is retrieved in top-k. |
| MRR | Measures how highly the first relevant document is ranked. |
| nDCG@10 | Measures ranking quality with higher reward for relevant documents near the top. |

Answer and grounding metrics:

| Metric | Purpose |
|---|---|
| Exact Match | Strict string-level answer match. |
| Token F1 | Token overlap between generated answer and gold answer. |
| ROUGE-L | Longest common subsequence overlap with gold answer. |
| Top-1 / Top-5 source hit | Whether the retrieved sources contain the gold source. |
| Citation accuracy | Whether the answer cites the expected source label. |
| Faithfulness proxy | Whether answer tokens are supported by retrieved context tokens. |
| NLI faithfulness judge | A local multilingual NLI model checks semantic support from context. |

For legal QA, source hit, citation accuracy, and faithfulness are especially important because a fluent answer is not enough if it is not grounded in a legal document.

The retrieval metrics answer the question: "Did the system find the right source?" The answer metrics answer the question: "Did the system produce the expected answer?" The grounding metrics answer the question: "Can the answer be trusted as source-supported?" All three are needed for the rubric's first scenario.

The project uses Exact Match, but Exact Match is not expected to be high for extractive or generative QA because correct answers may be phrased differently from the gold answer. Token F1 and ROUGE-L are more forgiving because they measure partial overlap. Source-hit and citation metrics are especially relevant because a legal answer can be phrased differently but still be useful if it cites the correct legal source.

Faithfulness is evaluated in two ways. The lexical proxy measures how much of the answer appears in the retrieved context. This is simple and reproducible, but it can overestimate support because matching words do not always imply semantic entailment. The NLI judge is stricter: it treats the retrieved context as a premise and the answer as a hypothesis, then estimates whether the premise supports the hypothesis. This is why the NLI faithfulness score is lower than the lexical proxy.

## 5. Base RAG System

The Base RAG system is:

```text
BM25 retrieval -> answer generator -> citation
```

BM25 is a lexical retrieval method. It is a strong baseline for legal data because legal questions often contain exact legal terms, article names, and source-specific wording.

Original 1,000-query retrieval benchmark:

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.947 | 0.975 | 0.863 | 0.890 |
| Dense multilingual MiniLM | 0.613 | 0.676 | 0.518 | 0.556 |
| Hybrid, dense weight 0.35 | 0.940 | 0.969 | 0.855 | 0.883 |

BM25 is the strongest retrieval baseline in this benchmark.

This result is reasonable for Turkish legal QA. Many legal questions use exact legal terms such as crime names, article numbers, procedural rights, or statute-specific phrases. BM25 is designed to reward term overlap, so it can be very competitive when the user's question and the relevant legal source share vocabulary.

Dense retrieval is still important to test because it can retrieve semantically similar passages even when exact words differ. However, dense retrieval depends heavily on the embedding model and on whether that model understands the legal domain and Turkish legal terminology. A generic multilingual model may not represent domain-specific legal distinctions well enough without careful adaptation.

QA evaluation with BM25 and extractive source-grounded answers on the 240-question gold benchmark:

| System | EM | Token F1 | ROUGE-L | Top-1 Source Hit | Top-5 Source Hit | Citation Accuracy | Faithfulness Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 + extractive answer | 0.363 | 0.799 | 0.793 | 0.813 | 0.908 | 0.813 | 0.961 |

This establishes the main reliable baseline.

The Base RAG system is intentionally simple: it retrieves with BM25 and uses an extractive source-grounded answer. This simplicity is a strength for legal QA because the output is easy to audit. The answer is not produced from model memory alone; it is directly tied to a retrieved source. The downside is that the answer can be less fluent or less synthesized than a generative LLM answer. The project treats this as a trade-off between fluency and legal reliability.

## 6. Fine-Tuned RAG Variants

Three fine-tuning directions were implemented and evaluated:

1. Fine-tuned embedding model
2. Fine-tuned cross-encoder reranker
3. Fine-tuned FLAN-T5 generator

The goal was not to assume fine-tuning always improves the system. The goal was to measure whether each fine-tuned component improves the same pipeline or the relevant ablation.

This is an important experimental principle. If a fine-tuned component is compared against a different pipeline, the result may not isolate the effect of fine-tuning. For example, comparing BM25 against a fine-tuned dense retriever does not only compare base versus fine-tuned; it also compares lexical retrieval against vector retrieval. Therefore, the report separates component-level ablations from final-system selection.

The fine-tuned components are evaluated as follows:

- Embedding ablation: base dense retriever versus fine-tuned dense retriever.
- Reranker ablation: pretrained cross-encoder reranker versus fine-tuned cross-encoder reranker.
- LLM ablation: base FLAN-T5-small versus fine-tuned FLAN-T5-small under the same retrieval setup.
- End-to-end submission check: Base BM25 RAG versus BM25 plus fine-tuned reranker on the same 150-question benchmark.

### 6.1 Fine-Tuned Embedding Model

The embedding model was fine-tuned with query-positive-negative triples:

```text
query, positive_passage, negative_passage
```

The intended effect is to move a legal question closer to its relevant legal passage in vector space and farther from irrelevant passages.

Full CPU fine-tuning used all 2,059 triples with one epoch, batch size 4, and max sequence length 256. Training took 2,471 seconds.

Dense retrieval before and after fine-tuning:

| Dense Model | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Base multilingual MiniLM | 0.613 | 0.676 | 0.518 | 0.556 |
| Fine-tuned embedding model | 0.539 | 0.591 | 0.456 | 0.488 |

Result: embedding fine-tuning did not improve dense retrieval on this benchmark. This is reported as a negative ablation result.

The negative result does not mean embedding fine-tuning is useless in general. It means that this particular fine-tuning setup did not improve this benchmark. Possible reasons include limited CPU training, negative sampling quality, small batch size, sequence length truncation, and mismatch between the embedding objective and the evaluation distribution. Reporting this result is still valuable because it prevents the final system from using a fine-tuned component just because it is more complex.

### 6.2 Fine-Tuned Reranker

The reranker pipeline is:

```text
Question -> BM25 candidates -> cross-encoder reranker -> top-k sources
```

A pretrained multilingual cross-encoder reranker was first tested, then fine-tuned using the reranker training data.

100-query subset:

| System | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 first stage on same 100 queries | 1.000 | 1.000 | 0.990 | 0.993 |
| Pretrained cross-encoder reranker | 0.700 | 0.810 | 0.550 | 0.612 |
| Fine-tuned cross-encoder reranker | 0.950 | 0.970 | 0.898 | 0.916 |

The fine-tuned reranker improves substantially over the pretrained reranker.

Full 1,000-query benchmark:

| System | Queries | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 first-stage ranking | 1,000 | 0.947 | 0.975 | 0.864 | 0.890 |
| Fine-tuned reranker | 1,000 | 0.882 | 0.915 | 0.789 | 0.820 |

Result: the fine-tuned reranker improves over the pretrained reranker, but BM25 remains stronger on the full benchmark.

This result shows two different conclusions at the same time. First, domain adaptation helps the reranker: the fine-tuned reranker is clearly better than the pretrained reranker. Second, the reranker does not automatically improve the entire RAG system when BM25 is already very strong. If the reranker changes the ordering of highly relevant BM25 results in the wrong direction, citation accuracy and top-1 source hit can decrease.

For this reason, the fine-tuned reranker is kept as an ablation result and optional component rather than the default live configuration.

### 6.3 Fine-Tuned FLAN-T5 Generator

For the LLM component, `google/flan-t5-small` was used because it is an open-source seq2seq model that can be fine-tuned and run locally on CPU for a small smoke experiment.

Training setup:

| Model | Train examples | Eval examples | Runtime | Train loss | Eval loss |
|---|---:|---:|---:|---:|---:|
| FLAN-T5-small SFT smoke | 448 | 64 | 629 sec | 7.368 | 0.514 |

The input format is source-grounded:

```text
question + retrieved/legal context -> target answer
```

Same-pipeline LLM comparison on a 20-question benchmark slice:

| Generator | Token F1 | ROUGE-L | Citation Accuracy | Faithfulness Proxy |
|---|---:|---:|---:|---:|
| Base FLAN-T5-small | 0.195 | 0.187 | 0.200 | 0.486 |
| Fine-tuned FLAN-T5-small | 0.232 | 0.210 | 0.350 | 0.593 |

Result: fine-tuning improves the same FLAN-T5 model under the same retrieval and generation pipeline. However, the fine-tuned model is still weaker than the extractive answer mode for legal citation reliability.

This is the most direct response to the "same LLM" requirement. The base and fine-tuned systems use the same model family, the same retrieval setup, and the same evaluation slice. The fine-tuned FLAN-T5 model improves over the base FLAN-T5 model on token F1, ROUGE-L, citation accuracy, and faithfulness proxy. However, the absolute citation score is still low for a legal QA demo. Therefore, the fine-tuned LLM is documented and available as an optional generation path, while the default demo uses the safer extractive answer generator.

## 7. Base RAG vs Fine-Tuned RAG Comparison

The instructor note asks for Base RAG and Fine-tuned RAG to be compared on the same benchmark. The project includes a runner for this:

```bash
python scripts/run_base_vs_finetuned_eval.py --data-dir DATA_DIR --output-dir OUTPUT_DIR
```

Latest submission-style comparison on the 150-question local benchmark:

Base RAG:

```text
BM25 retrieval -> extractive answer
```

Fine-tuned RAG:

```text
BM25 top-15 candidates -> fine-tuned cross-encoder reranker -> top-5 -> extractive answer
```

Both systems use the same corpus, same benchmark, and same extractive answer generator.

| Metric | Base BM25 RAG | Fine-tuned reranker RAG | Delta |
|---|---:|---:|---:|
| Token F1 | 0.703 | 0.631 | -0.072 |
| ROUGE-L | 0.702 | 0.621 | -0.080 |
| Top-1 source hit | 0.960 | 0.620 | -0.340 |
| Top-5 source hit | 0.993 | 0.900 | -0.093 |
| Citation accuracy | 0.960 | 0.620 | -0.340 |
| Faithfulness proxy | 0.928 | 0.934 | +0.006 |
| NLI faithfulness | 0.807 | 0.792 | -0.015 |

Interpretation: on this benchmark, the fine-tuned reranker variant does not improve the final RAG performance. It slightly increases lexical faithfulness, but it hurts source ranking and citation accuracy. Therefore, the final live demo keeps BM25 as the default retriever.

This result is important: the project does not hide negative fine-tuning results. It reports them and selects the final system based on measured performance.

The end-to-end comparison also clarifies the difference between a component improvement and a system improvement. The fine-tuned reranker improved over the pretrained reranker in the reranker-specific ablation, but the complete RAG pipeline with the fine-tuned reranker did not outperform the BM25 base system on the 150-question benchmark. Since the instructor's benchmark will evaluate the complete system, the final system choice must prioritize end-to-end metrics.

## 8. Ablation Study

The contribution of each fine-tuned component is summarized below.

| Component | Base System | Fine-Tuned Variant | Result | Final Decision |
|---|---|---|---|---|
| Embedding | Base multilingual dense retriever | Fine-tuned dense retriever | Worse Recall@10 and MRR | Not used in final demo |
| Reranker | Pretrained cross-encoder | Fine-tuned cross-encoder | Improved over pretrained reranker but did not beat BM25 | Reported as ablation |
| LLM | Base FLAN-T5-small | Fine-tuned FLAN-T5-small | Improved F1, citation, and faithfulness on 20-question slice | Optional mode only |
| Final demo | BM25 + extractive answer | N/A | Strongest citation-reliable live setup | Used as default demo |

Main conclusion from the ablation: fine-tuning is useful to test, but it does not automatically improve the final legal RAG system. Legal QA requires measured selection based on retrieval accuracy, citation correctness, and grounding.

This ablation design supports the final decision without relying on unsupported claims. The project can show exactly which fine-tuned component helped, which did not, and why the live system uses the configuration it uses. This is preferable to presenting a complex pipeline without evidence that each added component improves the final task.

## 9. Faithfulness and Hallucination-Oriented Evaluation

Hallucination in legal QA means producing an answer that is not supported by the retrieved legal source. For example, if the source says "muebbet hapis" but the model answers "20 years", that is an unsupported answer.

The system uses multiple grounding checks:

- Extractive answers to reduce unsupported generation risk
- Citation labels to show the source of the answer
- Lexical faithfulness proxy
- Local multilingual NLI faithfulness judge

NLI judge result on the 240-question original QA benchmark:

| Judge | Examples | Supported | Rejected | Faithfulness |
|---|---:|---:|---:|---:|
| Multilingual NLI judge | 240 | 206 | 34 | 0.858 |

This corresponds to an approximate unsupported-answer rate of:

```text
34 / 240 = 14.2%
```

The judge is stricter than lexical overlap, so this is a more conservative grounding estimate.

The approximate unsupported-answer rate is not presented as an absolute real-world hallucination rate. It is a benchmark-specific proxy based on the local NLI judge. Still, it is useful because it measures whether generated answers are semantically supported by retrieved sources. In legal QA, this is more important than surface fluency.

The extractive answer mode reduces hallucination risk because it does not freely generate legal claims from model memory. However, it can still produce an unsupported final answer if the retriever ranks the wrong source first. This is why retrieval quality and grounding quality are evaluated together.

## 10. Custom Document and Custom Benchmark Support

The instructor must be able to provide a custom document collection and a custom benchmark. This is supported through `--data-dir`.

Accepted custom corpus files:

```text
corpus.jsonl
corpus_index.jsonl
real_corpus.jsonl
```

Each corpus row must contain at least:

```json
{"id": "DOC_001", "text": "Legal source text"}
```

Optional fields:

```json
{"title": "Source title", "metadata": {"citation_label": "Citation label"}}
```

Accepted custom benchmark files:

```text
eval_qa.jsonl
custom_benchmark.jsonl
benchmark.jsonl
eval_qa_150.jsonl
```

Recommended benchmark row:

```json
{"question": "Kasten oldurme sucu nedir?", "gold_answer": "...", "source_id": "DOC_001"}
```

Validation command:

```bash
python scripts/validate_custom_data.py --data-dir sample_custom_data --require-benchmark
```

Demo on custom documents:

```bash
python scripts/demo_app.py --data-dir sample_custom_data
```

Evaluation on custom benchmark:

```bash
python scripts/run_base_vs_finetuned_eval.py --data-dir sample_custom_data --output-dir outputs/sample_submission_eval
```

This directly addresses the custom document and custom benchmark requirements.

The custom data support is designed for the instructor's evaluation workflow. If the instructor provides a new document collection, the collection can be converted into the accepted JSONL corpus format. If the instructor provides a new benchmark, the benchmark can be placed in the same folder using one of the accepted benchmark filenames. The same evaluation scripts can then run without changing the model code.

This is important because the final grade should not depend only on self-reported results. The system must be testable on an external document collection and an external benchmark. The validator script helps catch common format problems before evaluation, such as missing document IDs or benchmark source IDs that do not exist in the corpus.

## 11. Final System Choice

The final default demo system is:

```text
BM25 retrieval -> extractive source-grounded answer -> citation
```

This is not chosen because it is the most complex model. It is chosen because it is the most reliable measured configuration for live Turkish legal QA in this environment.

Reasons:

- BM25 had the strongest retrieval benchmark performance.
- Fine-tuned dense retrieval did not improve dense retrieval.
- Fine-tuned reranking improved over the pretrained reranker but did not beat BM25 end-to-end.
- Fine-tuned FLAN-T5 improved over base FLAN-T5 but still had weak citation reliability.
- Legal QA prioritizes grounded, auditable answers over fluent but unsupported generation.

The optional LLM generation path remains available:

```bash
python scripts/demo_app.py --data-dir data --answer-mode local_hf --generation-model outputs/models/flan_t5_legal_sft_smoke_512
```

However, the default demo remains extractive because it is more citation-reliable.

This final-system choice is aligned with the legal domain. A more fluent but weakly grounded LLM answer is less appropriate for legal QA than a more conservative answer with a clear source. The report therefore separates two claims:

- The project implements and evaluates fine-tuned LLM and reranker components.
- The deployed live configuration is selected by measured reliability, not by model complexity.

## 12. Reproducibility

Important commands:

```bash
python scripts/evaluate_retrieval.py --data-dir data --retriever bm25 --top-k 10
python scripts/evaluate_qa.py --data-dir data --retriever bm25 --generation-mode extractive --top-k 5
python scripts/run_base_vs_finetuned_eval.py --data-dir data --output-dir outputs/submission_eval_bm25_vs_finetuned_reranker --base-retriever bm25 --finetuned-retriever bm25 --finetuned-reranker-model PATH_TO_RERANKER
python scripts/validate_custom_data.py --data-dir sample_custom_data --require-benchmark
python scripts/demo_app.py --data-dir data
```

Main supporting documents:

| File | Purpose |
|---|---|
| `docs/submission_custom_evaluation_guide.md` | Custom corpus and benchmark instructions |
| `docs/submission_evaluation_results.md` | Latest submission-style evaluation results |
| `docs/controlled_ablation_summary.md` | Base vs fine-tuned component ablations |
| `docs/reproducibility_evidence.md` | Exact commands and measured outputs |
| `SUBMISSION_READY.md` | Short checklist for final submission |

## 13. Limitations

The project has several limitations:

- The default answer generator is extractive, not a full production-level fine-tuned LLM.
- The local machine is CPU-only; no CUDA GPU was available.
- FLAN-T5-small is a small model and is not strong enough for high-quality Turkish legal generation.
- Fine-tuned embedding and reranker components did not improve the final benchmark enough to replace BM25.
- The NLI faithfulness judge is a local semantic judge, not an external API-based LLM judge.

These limitations are reported explicitly because the project selection is based on measured behavior, not on assuming that every fine-tuned component improves performance.

Future work should focus on stronger GPU-based instruction tuning of a Turkish-capable legal LLM, better negative sampling for dense retrieval, and reranker validation against an external benchmark before deployment. A stronger generative model could eventually replace the extractive answer mode if it achieves comparable citation accuracy and faithfulness.

## 14. Conclusion

This project implements a Turkish Legal RAG system with custom document support, gold-benchmark evaluation, Base RAG vs Fine-tuned RAG comparisons, and ablation studies for embedding, reranker, and LLM components.

The most reliable default system is BM25 retrieval with extractive source-grounded answering and citation. The fine-tuned LLM improves over the base FLAN-T5 model in the same generation pipeline, satisfying the same-LLM comparison requirement, but it remains weaker than extractive answering for citation-critical legal QA.

The final conclusion is:

```text
Fine-tuned components were implemented and evaluated, but the deployed live configuration is selected by benchmark performance and citation reliability.
```

This conclusion directly follows the rubric: retrieval is evaluated because gold documents exist, answer quality is evaluated because gold answers exist, and grounding is evaluated because correct answers without support should be penalized in legal RAG.
