# Engineering Decisions

This document records the main design decisions made while building the RAG evaluation pipeline and the reasoning behind them.

The goal was to make decisions based on measurable experiments rather than assuming that a more complex technique would automatically perform better.

---

## 1. Use Official FastAPI Documentation

### Decision

Use the official FastAPI documentation as the knowledge base.

### Why

FastAPI documentation provides a focused technical domain with structured content, code examples, API names, parameters, decorators, and configuration details.

This makes it suitable for evaluating both semantic retrieval and exact lexical matching.

A controlled domain also makes it possible to create a reliable question-answer benchmark with known source URLs.

---

## 2. Use a Handwritten 50-Question Benchmark

### Decision

Create a domain-specific benchmark of 50 questions with expected answers, source URLs, key points, and selected factuality constraints.

### Why

A fixed benchmark provides a stable basis for comparing retrieval strategies.

Instead of evaluating the system only through qualitative examples, every retrieval configuration can be tested against the same questions.

The benchmark is intentionally small and domain-specific because the purpose is to compare engineering choices within a controlled environment.

---

## 3. Use `all-MiniLM-L6-v2` for Embeddings

### Decision

Use:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for document and query embeddings.

### Why

The project is designed to run locally and in GitHub Actions without requiring a paid embedding API.

MiniLM provides a lightweight embedding model with sufficiently useful semantic representations for this technical-documentation benchmark while keeping inference and setup relatively inexpensive.

Using a fixed local embedding model also makes experiments easier to reproduce.

---

## 4. Use Fixed-Size Chunking as the Baseline

### Decision

Use fixed-size chunks with:

```text
Chunk size: 1000 characters
Overlap: 150 characters
```

### Why

Fixed-size chunking provides a simple and predictable baseline.

It is easy to reproduce, inspect, and modify, making it useful as a reference point when evaluating more complex chunking strategies.

The final fixed-size pipeline produced 368 chunks from 29 documentation pages.

---

## 5. Evaluate Semantic Chunking Separately

### Decision

Implement semantic chunking as an experiment rather than making it the default pipeline.

### Why

Semantic chunking can potentially produce more coherent retrieval units, but increased complexity does not guarantee better retrieval.

On this benchmark, the semantic chunking configuration did not outperform the fixed-size approach.

Keeping it as an experiment makes the result useful rather than discarding it simply because it did not become the final approach.

---

## 6. Use ChromaDB for Vector Storage

### Decision

Use ChromaDB as the local persistent vector store.

### Why

The project needs a persistent vector index that can be rebuilt locally and inside CI.

ChromaDB provides a simple interface for storing embeddings and performing similarity search without requiring an external database service.

The generated ChromaDB directories are excluded from Git and rebuilt when needed.

---

## 7. Add a CrossEncoder Reranker

### Decision

Use:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

to rerank the initial retrieval candidates.

### Why

Vector retrieval is efficient for generating a candidate set, but the initial ranking may not always place the most relevant chunk first.

A CrossEncoder can compare the query and candidate document together and produce a more relevance-aware ranking.

This creates a two-stage retrieval architecture:

```text
Candidate Retrieval
        ↓
CrossEncoder Reranking
```

---

## 8. Do Not Deduplicate Before Reranking

### Decision

The final reranking pipeline does not perform document-level deduplication before reranking.

### Why

This was based on an experiment rather than a theoretical assumption.

The measured results were:

```text
Vector + dedup + reranker
Hit@1: 70%
Hit@3: 84%
Hit@5: 92%

Vector + reranker, no dedup
Hit@1: 90%
Hit@3: 96%
Hit@5: 96%
```

Deduplicating documents before reranking removed candidate chunks that could still provide useful ranking information.

Therefore, deduplication was not retained in the final reranking pipeline.

---

## 9. Add BM25 for Lexical Retrieval

### Decision

Combine vector retrieval with BM25.

### Why

Technical documentation contains many exact identifiers and terms where lexical matching can be valuable.

Examples include:

* API names
* Parameters
* Decorators
* Status codes
* Configuration names
* Python identifiers

Vector search captures semantic similarity, while BM25 provides exact term matching.

The two approaches therefore provide complementary retrieval signals.

---

## 10. Use Reciprocal Rank Fusion

### Decision

Combine vector and BM25 rankings using Reciprocal Rank Fusion (RRF).

The implementation uses:

```text
k = 60
```

### Why

The vector and BM25 systems produce separate ranked lists.

RRF provides a simple way to combine those rankings without requiring the scores from the two retrieval systems to be calibrated to the same numerical scale.

The resulting candidate ranking is then passed to the CrossEncoder reranker.

---

## 11. Use Hybrid Retrieval as the Final Retrieval Pipeline

### Decision

The final retrieval pipeline is:

```text
Vector Search
      +
BM25
      ↓
RRF Fusion
      ↓
CrossEncoder Reranking
```

### Why

The hybrid pipeline achieved:

```text
Hit@1: 88%
Hit@3: 98%
Hit@5: 98%
```

on the 50-question benchmark.

The main reason for retaining it was its strong top-k coverage and complementary lexical matching, rather than optimizing for Hit@1 alone.

The result is benchmark-specific and does not imply that hybrid retrieval will always outperform simpler approaches.

---

## 12. Use SmolLM2-360M for Generation

### Decision

Use:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

for answer generation.

### Why

The project is intended to be reproducible without depending on a paid hosted LLM API.

A lightweight local model makes it possible to execute the complete pipeline on a developer machine and in GitHub Actions.

The generation model was intentionally kept fixed while retrieval experiments were conducted so that retrieval changes could be evaluated under a controlled generation configuration.

---

## 13. Use Top 5 Retrieved Chunks

### Decision

Provide the top 5 reranked chunks to the generation model.

### Why

Generation experiments showed that increasing the context from 3 to 5 chunks improved several answer-quality signals.

The final configuration therefore uses the top 5 retrieved chunks.

Keeping this parameter fixed also makes later experiments easier to compare.

---

## 14. Use Deterministic Generation

### Decision

Use deterministic generation with sampling disabled.

### Why

The purpose of the evaluation is to compare system behavior rather than introduce random variation between runs.

Deterministic generation makes benchmark results easier to reproduce and investigate.

---

## 15. Use Multiple Answer-Quality Metrics

### Decision

Evaluate generated answers using:

* Keyword matching
* Semantic similarity
* Key-point coverage
* Lightweight factuality checks

### Why

No single automated metric captures answer quality completely.

Keyword matching can detect important terms, semantic similarity captures broader meaning, key-point evaluation checks required concepts, and factuality guards target known incorrect claims.

These metrics are treated as engineering signals rather than replacements for human evaluation.

---

## 16. Keep Factuality Checks Lightweight

### Decision

Use explicit rule-based factuality constraints for selected benchmark questions.

### Why

The goal was to catch known failure modes without introducing another large model or external evaluation service.

For selected questions, the benchmark specifies claims that should not appear in the answer.

This provides a simple regression signal for known hallucination patterns.

It is intentionally not presented as a complete factuality evaluator.

---

## 17. Rebuild the Vector Index Instead of Committing It

### Decision

Do not commit the generated ChromaDB database to Git.

Instead, rebuild it using:

```text
data/chunks.json
        ↓
src/reindex_fixed.py
        ↓
ChromaDB
```

### Why

Vector databases contain generated artifacts that can be reproduced from the committed chunk data and embedding model.

Keeping the generated index out of Git makes the repository smaller and ensures that the evaluation process tests whether the system can reconstruct its own state.

---

## 18. Use GitHub Actions for Evaluation

### Decision

Run the complete evaluation pipeline automatically in GitHub Actions.

### Why

Local execution alone does not prove that the project works from a clean environment.

The CI workflow:

1. Creates a fresh runner
2. Installs the project dependencies
3. Rebuilds the vector index
4. Runs the 50-question evaluation
5. Uploads the evaluation results

This provides a reproducibility check and creates a foundation for future regression testing.

---

## 19. Do Not Optimize Only for a Single Metric

### Decision

Evaluate retrieval using Hit@1, Hit@3, and Hit@5 rather than relying only on Hit@1.

### Why

Different retrieval configurations can behave differently depending on how many relevant candidates are required downstream.

For a RAG pipeline, having the relevant information somewhere in the top-k candidate set can still be useful because the reranker and generator operate on retrieved candidates.

Reporting multiple cutoffs provides a more complete picture of retrieval behavior.

---

## Summary

The final system was not selected because each individual component was assumed to be the best available option.

Instead, the pipeline evolved through controlled experiments:

```text
Simple Baseline
      ↓
Chunking Experiments
      ↓
Vector Retrieval
      ↓
Reranking Experiments
      ↓
BM25 + Vector Hybrid
      ↓
RRF Fusion
      ↓
CrossEncoder Reranking
      ↓
End-to-End Evaluation
      ↓
GitHub Actions Reproducibility
```

The main engineering lesson from the project is that **RAG quality depends on the interaction between retrieval, ranking, context selection, and generation**, so individual components should be evaluated within the complete pipeline rather than judged in isolation.
