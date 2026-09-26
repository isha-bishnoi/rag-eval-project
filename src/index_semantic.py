import json

from sentence_transformers import SentenceTransformer

from retrieval.vector_search import (
    SEMANTIC_CHUNKS_PATH,
    create_collection,
    index_chunks,
    MODEL_NAME,
)


def main():
    with open(SEMANTIC_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    collection = create_collection(
        name="fastapi_docs_semantic"
    )

    index_chunks(
        collection,
        chunks,
        model,
    )

    print(f"Indexed chunks: {collection.count()}")
    print("Collection: fastapi_docs_semantic")


if __name__ == "__main__":
    main()