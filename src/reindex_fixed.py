import chromadb

from sentence_transformers import SentenceTransformer

from retrieval.vector_search import (
    CHROMA_PATH,
    MODEL_NAME,
    load_chunks,
    create_collection,
    index_chunks,
)


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Remove the old fixed collection.
    try:
        client.delete_collection("fastapi_docs")
        print("Deleted old fastapi_docs collection.")
    except Exception:
        print("fastapi_docs collection did not exist.")

    # Create a fresh collection with chunk_id metadata.
    collection = create_collection(
        name="fastapi_docs"
    )

    chunks = load_chunks("data/chunks.json")

    model = SentenceTransformer(MODEL_NAME)

    index_chunks(
        collection,
        chunks,
        model,
    )

    print(f"Indexed chunks: {collection.count()}")


if __name__ == "__main__":
    main()