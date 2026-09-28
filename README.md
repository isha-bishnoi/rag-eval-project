# RAG Evaluation Pipeline

[![RAG Evaluation](https://github.com/isha-bishnoi/rag-eval-project/actions/workflows/eval.yml/badge.svg)](https://github.com/isha-bishnoi/rag-eval-project/actions/workflows/eval.yml)

A production-style Retrieval-Augmented Generation (RAG) evaluation pipeline built over the official FastAPI documentation.

The project focuses on **measuring and comparing RAG retrieval strategies**, rather than simply building a question-answering chatbot. It evaluates vector search, BM25, hybrid retrieval, reranking, chunking strategies, and generated answer quality using a controlled 50-question benchmark.

---

## Why This Project?

Many RAG applications demonstrate that a model can retrieve documents and generate an answer, but do not systematically measure whether the retrieval pipeline is actually improving.

This project treats RAG as an **engineering and evaluation problem**:

* How does chunking affect retrieval?
* Does reranking improve retrieval quality?
* Does combining semantic and lexical retrieval help with technical documentation?
* How often does the system retrieve the correct source?
* Does the generated answer contain the expected information?
* Can the entire evaluation be reproduced automatically in CI?

The goal is to answer these questions using measurable experiments.

---

## Architecture

```text
FastAPI Documentation
        │
        ▼
   Web Scraping
        │
        ▼
  Raw Documents
        │
        ▼
     Chunking
   ┌────┴─────┐
   │          │
 Fixed     Semantic
   │          │
   └────┬─────┘
        │
        ▼
    Embeddings
        │
        ▼
   Vector Search ─────────┐
                          │
   BM25 Search ───────────┤
                          ▼
                    Hybrid Retrieval
                          │
                          ▼
                       RRF Fusion
                          │
                          ▼
                    CrossEncoder
                     Reranking
                          │
                          ▼
                  Top-k Context
                          │
                          ▼
                   SmolLM2-360M
                          │
                          ▼
                   Generated Answer
                          │
                          ▼
                    Evaluation
              ┌───────────┴───────────┐
              │                       │
       Retrieval Metrics       Answer Metrics
       Hit@1 / Hit@3 / Hit@5   Keyword / Semantic
                                Key-point / Factuality
```

---

## Tech Stack

* **Python 3.11**
* **Sentence Transformers** — document embeddings
* **ChromaDB** — vector storage and retrieval
* **BM25** — lexical retrieval
* **CrossEncoder** — reranking
* **Hugging Face Transformers** — answer generation
* **SmolLM2-360M-Instruct** — lightweight local generation model
* **BeautifulSoup** — documentation extraction
* **GitHub Actions** — automated evaluation

---

## Data

The pipeline uses the official FastAPI documentation as its knowledge base.

The ingestion pipeline currently processes **29 documentation pages**, covering topics such as:

* Path and query parameters
* Pydantic models
* Request and response models
* Status codes
* Forms and files
* Exceptions
* Dependencies
* Security and OAuth2
* CORS
* Middleware
* Background tasks
* APIRouter
* Testing
* JSON encoding
* Body updates

The raw documentation is stored locally as structured JSON.

---

## Chunking Experiments

Two chunking strategies were implemented and compared.

### Fixed-size chunking

The baseline uses:

```text
Chunk size: 1000 characters
Overlap: 150 characters
```

This produced **368 chunks** from the 29 documentation pages.

### Semantic chunking

A semantic chunking experiment was also implemented to test whether grouping text based on semantic boundaries improved retrieval.

The experiment was retained as a comparison rather than automatically assuming that semantic chunking would be better.

---

## Retrieval Strategies

The project evaluates multiple retrieval configurations.

### 1. Vector Search

Documents are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and stored in ChromaDB.

### 2. Vector + Document Deduplication

Retrieved chunks are deduplicated at the document level before selecting the final results.

### 3. Vector + Reranking

Initial vector retrieval is followed by a CrossEncoder reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 4. Vector + Reranking Without Deduplication

This experiment tests whether deduplicating documents before reranking removes useful information.

### 5. Hybrid Retrieval

The final retrieval pipeline combines:

```text
Vector Search
      +
BM25
      ↓
Reciprocal Rank Fusion (RRF)
      ↓
CrossEncoder Reranking
```

BM25 provides exact lexical matching that can be useful for technical terms, API names, decorators, parameters, and status codes.

---

## Retrieval Results

The 50-question benchmark produced the following retrieval results:

| Retrieval Strategy                  |   Hit@1 |   Hit@3 |   Hit@5 |
| ----------------------------------- | ------: | ------: | ------: |
| Vector + document deduplication     |     70% |     94% |     98% |
| Vector + dedup + reranker           |     70% |     84% |     92% |
| Vector + reranker, no dedup         |     90% |     96% |     96% |
| Semantic chunks + reranker          |     82% |     94% |     94% |
| **Hybrid Vector + BM25 + reranker** | **88%** | **98%** | **98%** |

### What the experiments showed

The experiments highlighted several useful engineering trade-offs:

* Reranking improved retrieval when duplicate documents were not removed beforehand.
* Deduplication before reranking reduced retrieval performance in this benchmark.
* Hybrid retrieval improved the coverage of the top results compared with the basic vector pipeline.
* Lexical retrieval was useful for technical documentation where exact terms and API names matter.
* Semantic chunking did not outperform the fixed-size baseline in this benchmark.

The results are benchmark-specific and should not be interpreted as universal properties of these techniques.

---

## Evaluation Dataset

A handwritten benchmark of **50 FastAPI questions** was created.

Each question contains:

* Question ID
* Question
* Expected answer
* Expected source URL
* Key points
* Optional alternative acceptable key points
* Factuality constraints for selected questions

The benchmark is intentionally domain-specific so that retrieval and answer quality can be evaluated against known documentation sources.

---

## Answer Generation

The generation stage uses:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

The current configuration uses:

* Top 5 retrieved chunks
* Maximum 100 generated tokens
* Deterministic generation
* Context-only answering instructions

The model was intentionally kept lightweight so the complete pipeline can run locally and in GitHub Actions without requiring a paid hosted LLM API.

---

## Answer Evaluation

Generated answers are evaluated using several complementary metrics.

### Keyword Score

Measures whether important expected terms appear in the generated answer.

### Semantic Score

Measures semantic similarity between the generated answer and the expected answer.

### Key-point Score

Checks whether important concepts required by the benchmark answer are covered.

### Factuality Check

A lightweight rule-based check verifies that selected answers do not contain explicitly known incorrect claims.

These metrics are intended as engineering signals rather than replacements for human evaluation.

---

## Current End-to-End Results

The local evaluation currently reports:

```text
Questions:       50

Hit@1:           44/50  = 88%
Hit@3:           49/50  = 98%
Hit@5:           49/50  = 98%

Keyword:         48.59%
Semantic:        73.24%
Key-point:       50.83%
Factuality:      80%
```

The evaluation identified:

```text
Retrieval failure:  Q26
Factuality failure: Q49
```

The same retrieval metrics and failure IDs were reproduced in GitHub Actions. Some answer-quality metrics showed small variation between the local and CI environments.

---

## Reproducibility

The project does not commit the ChromaDB index to Git.

Instead, the vector index is rebuilt from the committed chunk data:

```text
data/chunks.json
        │
        ▼
src/reindex_fixed.py
        │
        ▼
ChromaDB index
        │
        ▼
src/eval.py
```

This makes it possible for a fresh environment to reconstruct the retrieval system.

---

## GitHub Actions

Evaluation is automatically executed through GitHub Actions.

The CI pipeline:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs dependencies
4. Rebuilds the ChromaDB index
5. Runs the 50-question evaluation
6. Uploads the evaluation results as an artifact

This verifies that the evaluation pipeline can run from a clean environment rather than depending on local state.

---

## Project Structure

```text
rag-eval-project/
│
├── data/
│   ├── raw/
│   ├── chunks.json
│   ├── semantic_chunks.json
│   └── qa_answer_key.json
│
├── src/
│   ├── ingest.py
│   ├── reindex_fixed.py
│   ├── eval.py
│   │
│   ├── chunking/
│   │   ├── fixed.py
│   │   └── semantic.py
│   │
│   └── retrieval/
│       ├── vector_search.py
│       ├── hybrid_search.py
│       └── reranker.py
│
├── results/
│   └── comparison_table.md
│
├── .github/
│   └── workflows/
│       └── eval.yml
│
├── requirements.txt
└── README.md
```

---

## Running Locally

Clone the repository and enter the project:

```bash
git clone https://github.com/isha-bishnoi/rag-eval-project.git
cd rag-eval-project
```

Set up Python 3.11:

pyenv install 3.11.11
pyenv local 3.11.11
python --version

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run ingestion:

```bash
python src/ingest.py
```

Create fixed-size chunks:

```bash
python src/chunking/fixed.py
```

Build the vector index:

```bash
python src/reindex_fixed.py
```

Run the evaluation:

```bash
python src/eval.py
```

Results are written to:

```text
results/evaluation_results.json
```

---

## Limitations

This project has several deliberate limitations:

* The benchmark contains 50 questions and focuses only on FastAPI documentation.
* The factuality evaluation uses lightweight rule-based checks.
* The generation model is intentionally small.
* Local CPU inference can be slow.
* CI execution requires downloading embedding, reranking, and generation models.
* Answer-quality metrics are automated proxies and are not equivalent to human evaluation.
* Retrieval results are specific to this dataset and benchmark.

---

## Future Improvements

Potential extensions include:

* Add retrieval latency and end-to-end latency measurements
* Track token usage and generation throughput
* Add more sophisticated factuality evaluation
* Add answer citations to retrieved documentation
* Add human evaluation for generated answers
* Add automated regression thresholds in CI
* Cache Hugging Face models in GitHub Actions
* Add unit tests for ingestion and retrieval components
* Experiment with stronger rerankers or generation models
* Add a small interactive demo on top of the evaluation pipeline

---

## Key Takeaway

This project demonstrates an end-to-end approach to **building, measuring, and reproducing a RAG system**.

The focus is not only on making a model answer questions, but on understanding how different retrieval and ranking decisions affect measurable performance and making those experiments reproducible through automated evaluation.
