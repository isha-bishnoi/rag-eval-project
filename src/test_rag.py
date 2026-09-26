
from sentence_transformers import SentenceTransformer

from retrieval.vector_search import (
    create_collection,
    retrieve,
    load_chunks,
)

from retrieval.hybrid_search import (
    create_bm25,
    keyword_search,
    reciprocal_rank_fusion,
)

from retrieval.reranker import (
    create_reranker,
    rerank,
)

from generate import (
    create_generator,
    generate_answer,
)


MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    question = "What is FastAPI used for?"

    # 1. Load embedding model
    embedding_model = SentenceTransformer(MODEL_NAME)

    # 2. Load vector database
    collection = create_collection(
        name="fastapi_docs"
    )

    # 3. Load chunks for BM25
    chunks = load_chunks(
        "data/chunks.json"
    )

    bm25 = create_bm25(chunks)

    # 4. Vector retrieval
    vector_results = retrieve(
        collection,
        embedding_model,
        question,
        top_k=15,
    )

    # 5. BM25 retrieval
    keyword_results = keyword_search(
        bm25,
        chunks,
        question,
        top_k=15,
    )

    # 6. Hybrid retrieval
    results = reciprocal_rank_fusion(
    vector_results,
    keyword_results,
    top_k=5,
)

    # 8. Show retrieved sources
    print("\nQuestion:")
    print(question)

    print("\nRetrieved Sources:")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   {result['url']}")

    # 9. Inspect retrieved context
    print("\nRetrieved Context:")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Chunk {i} ---")
        print(result["text"][:500])

    # 10. Load generator
    tokenizer, model = create_generator()

    # 11. Generate answer
    answer = generate_answer(
        tokenizer,
        model,
        question,
        results[1:2],
    )

    print("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()