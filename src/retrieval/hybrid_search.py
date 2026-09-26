import json
import re

from rank_bm25 import BM25Okapi


CHUNKS_PATH = "data/chunks.json"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def create_bm25(chunks):
    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(corpus)


def keyword_search(bm25, chunks, query, top_k=15):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            **chunks[index],
            "bm25_score": float(scores[index]),
        })

    return results

def reciprocal_rank_fusion(vector_results, keyword_results, top_k=15, k=60):
    scores = {}
    result_map = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        result_map[chunk_id] = result

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        result_map[chunk_id] = result

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:top_k]

    results = []

    for chunk_id in ranked_ids:
        results.append({
            **result_map[chunk_id],
            "rrf_score": scores[chunk_id],
        })

    return results