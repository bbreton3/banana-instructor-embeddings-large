from potassium import Potassium, Request, Response

from InstructorEmbedding import INSTRUCTOR
import torch
import time

app = Potassium("my_app")


# @app.init runs at startup, and initializes the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    MODEL_NAME = "hkunlp/instructor-large"

    model = INSTRUCTOR(MODEL_NAME)

    context = {"model": model}

    return context


# @app.handler is an http post handler running for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")

    # Start timer
    t_1 = time.time()

    # This allows for batched inference
    text_instruction_pairs = request.json.get("text_instruction_pairs")

    # postprocess
    texts_with_instructions = []
    for pair in text_instruction_pairs:
        texts_with_instructions.append([pair["instruction"], pair["text"]])

    # calculate embeddings
    customized_embeddings = model.encode(texts_with_instructions).tolist()

    t_2 = time.time()

    return Response(
        json={
            "customized_embeddings": customized_embeddings,
            "text_instruction_pairs": text_instruction_pairs,
            "inference_time": t_2 - t_1,
        },
        status=200,
    )


if __name__ == "__main__":
    app.serve()
