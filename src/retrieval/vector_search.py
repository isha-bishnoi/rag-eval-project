import json
import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/chunks.json"
SEMANTIC_CHUNKS_PATH = "data/semantic_chunks.json"
CHROMA_PATH = "data/chroma"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_collection(name="fastapi_docs"):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=name
    )

    return collection


def index_chunks(collection, chunks, model):
    documents = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "url": chunk["url"],
            }
            for chunk in chunks
        ],
    )


def retrieve(collection, model, query, top_k=5):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved_chunks = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved_chunks.append({
    "chunk_id": metadata["chunk_id"],
    "text": document,
    "title": metadata["title"],
    "url": metadata["url"],
    "distance": distance,
})

    return retrieved_chunks