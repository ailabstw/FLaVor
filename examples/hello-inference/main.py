from flavor.serve.apps import InferAPP
from flavor.serve.strategies import AiCOCOInputStrategy


def infer(images):

    # Your model & inference code

    return images  # demo code only, please return your output as defined in Readme


app = InferAPP(infer_function=infer, input_strategy=AiCOCOInputStrategy, output_strategy=None)

app.run()
