from flavor.serve.apps import InferAPP
from flavor.serve.strategies import AiCOCOInputStrategy


def infer(images):
    return images


app = InferAPP(infer_function=infer, input_strategy=AiCOCOInputStrategy(), output_strategy=None)

app.run()
