
import json

from sentence_transformers import SentenceTransformer

from retrieval.vector_search import (
    create_collection,
    retrieve,
    load_chunks,
)

from retrieval.reranker import (
    create_reranker,
    rerank,
)

from retrieval.hybrid_search import (
    create_bm25,
    keyword_search,
    reciprocal_rank_fusion,
)

from generate import (
    create_generator,
    generate_answer,
)


QA_PATH = "data/qa_answer_key.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_questions():
    with open(QA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return set(
        text.lower()
        .replace(",", "")
        .replace(".", "")
        .split()
    )


def answer_keyword_score(generated, expected):
    generated_words = normalize_text(generated)
    expected_words = normalize_text(expected)

    if not expected_words:
        return 0.0

    matched = generated_words & expected_words

    return len(matched) / len(expected_words)


def answer_key_point_score(generated, key_points):
    if not key_points:
        return None

    generated_text = generated.lower()

    matched = 0

    for alternatives in key_points:
        for phrase in alternatives:
            if phrase.lower() in generated_text:
                matched += 1
                break

    return matched / len(key_points)


def answer_factuality_score(generated, must_not_contain):
    if not must_not_contain:
        return None

    generated_text = generated.lower()

    for phrase in must_not_contain:
        if phrase.lower() in generated_text:
            return 0.0

    return 1.0


def answer_semantic_score(model, generated, expected):
    embeddings = model.encode(
        [generated, expected],
        normalize_embeddings=True,
    )

    return float(embeddings[0] @ embeddings[1])


def evaluate():
    questions = load_questions()

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    collection = create_collection(
        name="fastapi_docs"
    )

    chunks = load_chunks(
        "data/chunks.json"
    )

    bm25 = create_bm25(chunks)

    reranker = create_reranker()

    tokenizer, generator = create_generator()

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    answer_scores = []
    semantic_scores = []
    key_point_scores = []
    answer_results = []

    failed_questions = []

    for question in questions:

        # Vector retrieval
        vector_results = retrieve(
            collection,
            embedding_model,
            question["question"],
            top_k=15,
        )

        # BM25 retrieval
        keyword_results = keyword_search(
            bm25,
            chunks,
            question["question"],
            top_k=15,
        )

        # Hybrid retrieval
        results = reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            top_k=15,
        )

        # Reranking
        results = rerank(
            reranker,
            question["question"],
            results,
            top_k=5,
        )

        # Retrieval evaluation
        expected_url = question["source_url"]

        retrieved_urls = [
            result["url"]
            for result in results
        ]

        if expected_url in retrieved_urls[:1]:
            hit_at_1 += 1

        if expected_url in retrieved_urls[:3]:
            hit_at_3 += 1

        if expected_url in retrieved_urls[:5]:
            hit_at_5 += 1
        else:
            failed_questions.append({
                "id": question["id"],
                "question": question["question"],
                "expected_url": expected_url,
                "results": results,
            })

        # Generate answer
        generated_answer = generate_answer(
            tokenizer,
            generator,
            question["question"],
            results,
        )

        # Evaluate answer
        answer_score = answer_keyword_score(
            generated_answer,
            question["answer"],
        )

        semantic_score = answer_semantic_score(
            embedding_model,
            generated_answer,
            question["answer"],
        )

        # Key-point evaluation
        key_point_score = None

        if "key_points" in question:
            key_point_score = answer_key_point_score(
                generated_answer,
                question["key_points"],
            )

        # Factuality evaluation
        factuality_score = answer_factuality_score(
            generated_answer,
            question.get("must_not_contain"),
        )

        if key_point_score is not None:
            key_point_scores.append(key_point_score)

        answer_scores.append(answer_score)
        semantic_scores.append(semantic_score)

        answer_results.append({
            "id": question["id"],
            "question": question["question"],
            "expected": question["answer"],
            "generated": generated_answer,
            "keyword_score": answer_score,
            "semantic_score": semantic_score,
            "key_point_score": key_point_score,
            "factuality_score": factuality_score,
        })

    total = len(questions)

    print("\nHybrid Retrieval Evaluation")
    print("---------------------------")

    print(f"Questions: {total}")

    print(
        f"Hit@1: {hit_at_1}/{total} "
        f"({hit_at_1 / total:.2%})"
    )

    print(
        f"Hit@3: {hit_at_3}/{total} "
        f"({hit_at_3 / total:.2%})"
    )

    print(
        f"Hit@5: {hit_at_5}/{total} "
        f"({hit_at_5 / total:.2%})"
    )

    average_answer_score = (
        sum(answer_scores)
        / len(answer_scores)
    )

    average_semantic_score = (
        sum(semantic_scores)
        / len(semantic_scores)
    )

    average_key_point_score = (
        sum(key_point_scores) / len(key_point_scores)
        if key_point_scores
        else 0.0
    )

    factuality_scores = [
        result["factuality_score"]
        for result in answer_results
        if result["factuality_score"] is not None
    ]

    average_factuality_score = (
        sum(factuality_scores) / len(factuality_scores)
        if factuality_scores
        else 0.0
    )

    print(
        f"Average Answer Keyword Score: "
        f"{average_answer_score:.2%}"
    )

    print(
        f"Average Answer Semantic Score: "
        f"{average_semantic_score:.2%}"
    )

    print(
        f"Average Key-Point Score: "
        f"{average_key_point_score:.2%}"
    )

    print(
        f"Average Factuality Check: "
        f"{average_factuality_score:.2%}"
    )

    print("\nKey-Point Evaluation")
    print("--------------------")

    for result in answer_results:
        if result["key_point_score"] is not None:
            print(f"\nID: {result['id']}")
            print(f"Generated: {result['generated']}")

            print(
                f"Key points: "
                f"{questions[result['id'] - 1]['key_points']}"
            )

            print(
                f"Key-point score: "
                f"{result['key_point_score']:.2%}"
            )

    print("\nGenerated Answers")
    print("-----------------")

    for result in answer_results:

        print(f"\nID: {result['id']}")
        print(f"Question: {result['question']}")
        print(f"Expected: {result['expected']}")
        print(f"Generated: {result['generated']}")

        print(
            f"Keyword Score: "
            f"{result['keyword_score']:.2%}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score']:.2%}"
        )

        if result["key_point_score"] is not None:
            print(
                f"Key-Point Score: "
                f"{result['key_point_score']:.2%}"
            )

        if result["factuality_score"] is not None:
            print(
                f"Factuality Check: "
                f"{result['factuality_score']:.2%}"
            )

    print("\nFailed Factuality Checks")
    print("------------------------")

    for result in answer_results:
        if result["factuality_score"] == 0:
            print(f"\nID: {result['id']}")
            print(f"Question: {result['question']}")
            print(f"Generated: {result['generated']}")

    print("\nFailed Questions")
    print("----------------")

    for failure in failed_questions:

        print(f"\nID: {failure['id']}")

        print(
            f"Question: "
            f"{failure['question']}"
        )

        print(
            f"Expected: "
            f"{failure['expected_url']}"
        )

        print("\nRetrieved:")

        for i, result in enumerate(
            failure["results"],
            start=1,
        ):
            print(f"{i}. {result['url']}")

            print(
                f"   Title: "
                f"{result['title']}"
            )

            print(
                f"   Rerank score: "
                f"{result['rerank_score']:.4f}"
            )

    evaluation_results = {
        "benchmark": {
            "questions": total,
            "embedding_model": MODEL_NAME,
        },
        "generation_config": {
            "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "max_new_tokens": 100,
            "top_k_chunks": 5,
        },
        "retrieval": {
            "hit_at_1": hit_at_1 / total,
            "hit_at_3": hit_at_3 / total,
            "hit_at_5": hit_at_5 / total,
        },
        "generation": {
            "average_keyword_score": average_answer_score,
            "average_semantic_score": average_semantic_score,
            "average_key_point_score": average_key_point_score,
            "average_factuality_score": average_factuality_score,
        },
        "failed_retrieval_questions": [
            failure["id"]
            for failure in failed_questions
        ],
        "failed_factuality_questions": [
            result["id"]
            for result in answer_results
            if result["factuality_score"] == 0
        ],
    }

    with open(
        "results/evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            evaluation_results,
            f,
            indent=2,
        )

    print("\nEvaluation results saved to results/evaluation_results.json")

if __name__ == "__main__":
    evaluate()
