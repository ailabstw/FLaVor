import os
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np

from chexpert.utils.wrappers import Wrapper
from flavor.serve.apps import InferAPP
from flavor.serve.inference import (
    BaseAiCOCOInferenceModel,
    BaseAiCOCOInputDataModel,
    BaseAiCOCOOutputDataModel,
)
from flavor.serve.models import AiImage, InferCategory
from flavor.serve.strategies import AiCOCOClassificationOutputStrategy


class ClassificationInferenceModel(BaseAiCOCOInferenceModel):
    def __init__(self):
        self.formatter = AiCOCOClassificationOutputStrategy()

        self.thesholds = {
            "Atelectasis": 0.3,
            "Cardiomegaly": 0.04,
            "Consolidation": 0.17,
            "Edema": 0.12,
            "Enlarged Cardiomediastinum": 0.09,
            "Fracture": 0.07,
            "Lung Lesion": 0.05,
            "Lung Opacity": 0.26,
            "No Finding": 0.06,
            "Pleural Effusion": 0.14,
            "Pleural Other": 0.02,
            "Pneumonia": 0.04,
            "Pneumothorax": 0.31,
            "Support Devices": 0.49,
        }
        super().__init__()

    def define_inference_network(self):
        return Wrapper(os.path.join(os.getcwd(), "chexpert/instances/optimized_model.h5"))

    def set_categories(self):
        categories = [
            {"name": "Atelectasis"},
            {"name": "Cardiomegaly"},
            {"name": "Consolidation"},
            {"name": "Edema"},
            {"name": "Enlarged Cardiomediastinum"},
            {"name": "Fracture"},
            {"name": "Lung Lesion"},
            {"name": "Lung Opacity"},
            {"name": "No Finding"},
            {"name": "Pleural Effusion"},
            {"name": "Pleural Other"},
            {"name": "Pneumonia"},
            {"name": "Pneumothorax"},
            {"name": "Support Devices"},
        ]
        return categories

    def set_regressions(self):
        return None

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[np.ndarray, None, None]:
        img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)

        return img, None, None

    def inference(self, x: np.ndarray) -> np.ndarray:
        return self.network.predict(x)

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        categories: List[InferCategory],
        **kwargs
    ) -> Any:
        format_output = np.zeros(len(categories))
        for i, category in enumerate(categories):
            name = category["name"]
            format_output[i] = int(model_out[name] > self.thesholds[name])

        output = self.formatter(model_out=format_output, images=images, categories=categories)
        return output


app = InferAPP(
    infer_function=ClassificationInferenceModel(),
    input_data_model=BaseAiCOCOInputDataModel,
    output_data_model=BaseAiCOCOOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
