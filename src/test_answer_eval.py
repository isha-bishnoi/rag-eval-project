from eval import answer_keyword_score


expected = "FastAPI is a modern web framework for building APIs with Python."

generated = "FastAPI is a Python framework used for building APIs."

score = answer_keyword_score(
    generated,
    expected,
)

print(f"Score: {score:.2%}")