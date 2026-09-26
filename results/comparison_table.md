# Retrieval and Generation Evaluation Results

## Benchmark

Dataset: 50 FastAPI documentation questions

The benchmark evaluates both:

1. **Retrieval quality** — whether the expected documentation source is retrieved.
2. **Answer quality** — whether the generated answer matches the expected answer and required key points.

---

## Retrieval Evaluation

| Approach                          | Hit@1 | Hit@3 | Hit@5 |
| --------------------------------- | ----: | ----: | ----: |
| Vector + document deduplication   |   70% |   94% |   98% |
| Vector + deduplication + reranker |   70% |   84% |   92% |
| Vector chunks + reranker          |   90% |   96% |   96% |
| Semantic chunks + reranker        |   82% |   94% |   94% |
| Hybrid (Vector + BM25) + reranker |   88% |   98% |   98% |

### Retrieval Observations

* Document-level deduplication before reranking reduced retrieval performance.
* Removing deduplication allowed the reranker to compare multiple chunks from the same document.
* Cross-encoder reranking improved Hit@1 from 70% to 90% compared with vector retrieval with document-level deduplication.
* The final hybrid pipeline retrieves the expected source for 49/50 benchmark questions at Hit@5.
* Q26 remains unresolved at Hit@5 because several dependency-related documentation pages are highly similar.
* Semantic chunking reduced the total number of chunks from 368 to 306.
* Despite producing larger structure-aware chunks, semantic chunking performed worse than fixed-size chunking on this benchmark.
* The decrease was observed across Hit@1, Hit@3, and Hit@5.
* For this dataset, fixed-size chunking currently provides stronger retrieval performance than semantic chunking.
* Hybrid retrieval improved Hit@3 and Hit@5 from 96% to 98%.
* Hit@1 decreased slightly from 90% to 88%.
* BM25 provides complementary exact-term matching for technical queries.
* The hybrid pipeline reduced the number of benchmark questions missing the expected source from 2 to 1 at Hit@5.

---

## Final Retrieval Pipeline

The current retrieval pipeline is:

```text
User Question
      ↓
Vector Search
      +
BM25 Keyword Search
      ↓
Reciprocal Rank Fusion (RRF)
      ↓
Cross-Encoder Reranker
      ↓
Top 5 Chunks
```

Final retrieval performance:

| Metric | Result |
| ------ | -----: |
| Hit@1  |    88% |
| Hit@3  |    98% |
| Hit@5  |    98% |

---

## Generation Evaluation

The final generation experiment uses:

* **Generator:** `HuggingFaceTB/SmolLM2-360M-Instruct`
* **Retrieved context:** Top 5 reranked chunks
* **Maximum new tokens:** 100
* **Benchmark size:** 50 questions

| Metric                        | Result |
| ----------------------------- | -----: |
| Average Answer Keyword Score  | 48.59% |
| Average Answer Semantic Score | 73.24% |
| Average Key-Point Score       | 50.83% |
| Average Factuality Check      | 80.00% |

### Generation Observations

* Increasing the generation limit from 60 to 100 tokens improved the Average Answer Keyword Score from 45.31% to 48.59%.
* Average Key-Point Score increased from 40.00% to 50.83%.
* Average Semantic Score remained approximately unchanged at 73.24%.
* The simple factuality check improved from 60% to 80%.
* Retrieval performance remained unchanged because only the generation configuration was modified.
* The smaller generation model sometimes produced technically incorrect answers despite receiving relevant documentation context.
* Common generation errors included confusing path parameters with query parameters, misinterpreting `Depends`, confusing application metadata with OpenAPI tag metadata, and incorrectly explaining `exclude_unset`.
* These results indicate that generation quality is currently a larger limitation than retrieval quality.
* The current project therefore freezes the retrieval pipeline rather than adding additional retrieval techniques.

---

## Controlled Generation Experiment

The effect of increasing the context/output budget was evaluated while keeping retrieval and the generation model unchanged.

| Configuration                            |    Keyword | Semantic |  Key-Point | Factuality |
| ---------------------------------------- | ---------: | -------: | ---------: | ---------: |
| Top 3 chunks, 60 tokens                  |     42.91% |   75.05% |     39.17% |        40% |
| Top 3 chunks, 60 tokens + refined prompt |     44.61% |   76.26% |     35.83% |        60% |
| Top 5 chunks, 60 tokens                  |     45.31% |   73.23% |     40.00% |        80% |
| Top 5 chunks, 100 tokens                 | **48.59%** |   73.24% | **50.83%** |        80% |

> Note: The first row represents the initial prompt configuration before the prompt refinement experiment.

---

## Key Findings

### Retrieval

The experiments show that:

* Fixed-size chunking performed better than semantic chunking on this benchmark.
* Cross-encoder reranking substantially improved top-ranked retrieval quality.
* Removing document-level deduplication before reranking improved performance because the reranker could compare multiple chunks from the same source.
* Hybrid retrieval provided better Hit@3 and Hit@5 performance than vector retrieval with reranking alone.
* The final retrieval pipeline achieves 98% Hit@3 and 98% Hit@5 on the 50-question benchmark.

### Generation

The experiments show that:

* Increasing the generation limit from 60 to 100 tokens improved answer completeness.
* The generation model remains the primary limitation in the current system.
* Some answers contain technically incorrect details even when relevant documentation was retrieved.
* Retrieval-source correctness and answer correctness are therefore treated as separate evaluation dimensions.

### Current Limitation

The benchmark contains 50 manually created questions, so the reported results are specific to this dataset and FastAPI documentation corpus.

The current evaluation also uses lightweight automated answer metrics rather than a full LLM-as-a-judge or human evaluation process. The factuality metric is a simple guard against predefined incorrect phrases and should therefore not be interpreted as a complete factuality assessment.

---

## Current Status

The current system consists of:

```text
FastAPI Documentation
        ↓
Document Ingestion
        ↓
Fixed-Size Chunking
        ↓
Vector Embeddings
        ↓
        ┌───────────────┐
        │ Vector Search │
        │      +        │
        │     BM25      │
        └───────┬───────┘
                ↓
        Reciprocal Rank Fusion
                ↓
        Cross-Encoder Reranker
                ↓
           Top 5 Chunks
                ↓
      SmolLM2-360M-Instruct
                ↓
          Generated Answer
                ↓
        Automated Evaluation
```

The retrieval pipeline is currently frozen at:

**Hybrid Vector + BM25 → RRF → Cross-Encoder Reranker → Top 5 chunks**

with:

* Hit@1: **88%**
* Hit@3: **98%**
* Hit@5: **98%**

The generation pipeline is currently frozen at:

* SmolLM2-360M-Instruct
* Top 5 retrieved chunks
* 100 maximum output tokens
