import json

RAW_PATH = "data/raw"
OUTPUT_PATH = "data/semantic_chunks.json"

TARGET_SIZE = 1000


def split_into_units(text):
    """
    Split document text into paragraph-like units.
    Blank lines are treated as boundaries.
    """
    units = [
        unit.strip()
        for unit in text.split("\n\n")
        if unit.strip()
    ]

    return units


def create_semantic_chunks(documents):
    all_chunks = []

    for document in documents:
        units = split_into_units(document["text"])

        current_chunk = ""

        for unit in units:
            # If adding this unit stays within the target size,
            # keep building the current chunk.
            if len(current_chunk) + len(unit) + 2 <= TARGET_SIZE:
                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += unit

            else:
                # Save the current chunk before starting a new one.
                if current_chunk:
                    all_chunks.append({
                        "text": current_chunk,
                        "title": document["title"],
                        "url": document["url"],
                    })

                # Keep an oversized unit intact rather than
                # splitting it in the middle.
                current_chunk = unit

        # Save the final chunk.
        if current_chunk:
            all_chunks.append({
                "text": current_chunk,
                "title": document["title"],
                "url": document["url"],
            })

    return all_chunks


def main():
    documents = []

    # Load the raw scraped documents.
    import os

    for filename in os.listdir(RAW_PATH):
        if filename.endswith(".json"):
            with open(
                os.path.join(RAW_PATH, filename),
                "r",
                encoding="utf-8",
            ) as f:
                documents.append(json.load(f))

    chunks = create_semantic_chunks(documents)

    # Add unique IDs.
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"semantic-{index}"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Documents: {len(documents)}")
    print(f"Semantic chunks: {len(chunks)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()