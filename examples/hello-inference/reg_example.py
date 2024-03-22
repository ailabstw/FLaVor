import os
from typing import List, Sequence, Tuple

import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from flavor.serve.apps import InferAPP
from flavor.serve.inference import BaseInferenceModel
from flavor.serve.models import InferRegressionOutput
from flavor.serve.strategies import AiCOCOInputStrategy, AiCOCORegressionOutputStrategy


class RegressionInferenceModel(BaseInferenceModel):
    def __init__(self, output_data_model: InferRegressionOutput):
        super().__init__(output_data_model=output_data_model)

    def define_inference_network(self):
        network = resnet18(ResNet18_Weights.DEFAULT)
        network.fc = nn.Linear(512, 2)
        network = network.eval()
        return network

    def define_categories(self):
        return None

    def define_regressions(self):
        regressions = {
            0: {"name": "reg0"},
            1: {"name": "reg1"},
        }
        return regressions

    def preprocess(self, data_filenames: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        img = Image.open(data_filenames[0])

        transforms = ResNet18_Weights.DEFAULT.transforms()
        img = transforms(img).unsqueeze(0)

        return img, data_filenames

    def postprocess(self, model_out: np.ndarray) -> np.ndarray:
        return model_out.squeeze(0)


if __name__ == "__main__":

    app = InferAPP(
        infer_function=RegressionInferenceModel(output_data_model=InferRegressionOutput),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCORegressionOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9999)))
