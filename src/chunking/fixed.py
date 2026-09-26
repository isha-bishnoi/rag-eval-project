import json
import glob
from pathlib import Path
from urllib.parse import urlparse

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_documents():
    documents = []

    for file_path in glob.glob("data/raw/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            document = json.load(f)

        documents.append(document)

    return documents


def create_chunks(documents):
    all_chunks = []

    for document in documents:
        chunks = chunk_text(document["text"])

        path = urlparse(document["url"]).path.strip("/")
        doc_id = path.replace("/", "-")

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc_id}-{index}",
                "text": chunk,
                "title": document["title"],
                "url": document["url"],
            })

    return all_chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = create_chunks(documents)

    output_path = Path("data/chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Saved: {output_path}")