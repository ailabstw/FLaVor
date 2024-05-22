import os
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from flavor.serve.apps import InferAPP
from flavor.serve.inference import (
    BaseAiCOCOInferenceModel,
    BaseAiCOCOInputDataModel,
    BaseAiCOCOOutputDataModel,
)
from flavor.serve.models import AiImage, InferRegression
from flavor.serve.strategies import AiCOCORegressionOutputStrategy


class RegressionInferenceModel(BaseAiCOCOInferenceModel):
    def __init__(self):
        self.formatter = AiCOCORegressionOutputStrategy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    def define_inference_network(self):
        network = resnet18(ResNet18_Weights.DEFAULT)
        network.fc = nn.Linear(512, 2)
        network.eval()
        network.to(self.device)
        return network

    def set_categories(self):
        return None

    def set_regressions(self):
        regressions = [
            {"name": "reg0"},
            {"name": "reg1"},
        ]
        return regressions

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[np.ndarray, None, None]:
        img = Image.open(files[0])
        return img, None, None

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        transforms = ResNet18_Weights.DEFAULT.transforms()
        img = transforms(data).unsqueeze(0).to(self.device)
        return img

    def postprocess(self, model_out: torch.Tensor, **kwargs) -> np.ndarray:
        return model_out.squeeze(0).cpu().detach().numpy()

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        regressions: List[InferRegression],
        **kwargs
    ) -> Any:
        output = self.formatter(model_out=model_out, images=images, regressions=regressions)
        return output


app = InferAPP(
    infer_function=RegressionInferenceModel(),
    input_data_model=BaseAiCOCOInputDataModel,
    output_data_model=BaseAiCOCOOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
