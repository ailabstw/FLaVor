import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    AiCOCOImageInputDataModel,
    AiCOCOImageOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiImage
from flavor.serve.inference.inference_models import BaseAiCOCOImageInferenceModel
from flavor.serve.inference.strategies import AiCOCORegressionOutputStrategy

torch.manual_seed(1234)
np.random.seed(1234)


class RegressionInferenceModel(BaseAiCOCOImageInferenceModel):
    def __init__(self):
        super().__init__()
        self.formatter = AiCOCORegressionOutputStrategy()

    def define_inference_network(self) -> Callable:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = resnet18(ResNet18_Weights.DEFAULT)
        network.fc = nn.Linear(512, 2)
        network.eval()
        network.to(self.device)
        return network

    def set_categories(self) -> None:
        return None

    def set_regressions(self) -> List[Dict[str, Any]]:
        regressions = [
            {"name": "reg0"},
            {"name": "reg1"},
        ]
        return regressions

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[Image.Image, None]:
        img = Image.open(files[0])
        return img, None

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        transforms = ResNet18_Weights.DEFAULT.transforms()
        img = transforms(data).unsqueeze(0).to(self.device)
        return img

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.network(x)
        return out

    def postprocess(self, model_out: torch.Tensor, **kwargs) -> np.ndarray:
        return model_out.squeeze(0).cpu().detach().numpy()

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        regressions: Sequence[Dict[str, Any]],
        **kwargs
    ) -> AiCOCOImageOutputDataModel:
        output = self.formatter(model_out=model_out, images=images, regressions=regressions)
        return output


app = InferAPP(
    infer_function=RegressionInferenceModel(),
    input_data_model=AiCOCOImageInputDataModel,
    output_data_model=AiCOCOImageOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
