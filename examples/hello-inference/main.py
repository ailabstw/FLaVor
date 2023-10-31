import os

from flavor.serve.apps import InferAPP
from flavor.serve.strategies import AiCOCOInputStrategy


def infer(**kwargs):
    return kwargs.get("images")


app = InferAPP(infer_function=infer, input_strategy=AiCOCOInputStrategy, output_strategy=None)
app.run(port=int(os.getenv("PORT", 9000)))
