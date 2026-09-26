from sentence_transformers import SentenceTransformer

from retrieval.vector_search import (
    load_chunks,
    create_collection,
    index_chunks,
    MODEL_NAME,
)


def main():
    chunks = load_chunks("data/chunks.json")

    model = SentenceTransformer(MODEL_NAME)

    collection = create_collection(
        name="fastapi_docs"
    )

    index_chunks(
        collection,
        chunks,
        model,
    )

    print(f"Indexed chunks: {collection.count()}")


if __name__ == "__main__":
    main()