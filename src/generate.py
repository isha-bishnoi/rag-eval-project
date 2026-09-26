from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


def create_generator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    return tokenizer, model


def build_prompt(question, context):
    messages = [
        {
            "role": "system",
            "content": (
    "You are a FastAPI documentation question answering system. "
    "Answer the user's question using only the provided documentation. "
    "Give a direct, complete answer to the question. "
    "Include the important technical names, parameters, decorators, "
    "status codes, or concepts needed to answer it correctly. "
    "If the question asks how something is done, include the relevant "
    "FastAPI API, function, parameter, or code pattern from the documentation. "
    "Do not repeat the question. "
    "Do not write 'Question:' or 'Answer:'. "
    "Do not copy long passages from the documentation. "
    "Do not invent APIs, parameters, decorators, or behavior. "
    "Keep the answer concise, using at most 2 sentences."
),
        },
        {
            "role": "user",
            "content": (
                f"Documentation:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Give the direct answer:"
            ),
        },
    ]

    return messages


def generate_answer(tokenizer, model, question, results):
    context = "\n\n---\n\n".join(
        result["text"]
        for result in results[:5]
    )

    messages = build_prompt(
        question,
        context,
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        do_sample=False,
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return answer.strip()