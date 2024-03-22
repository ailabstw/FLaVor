import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from base_inference_model import BaseInferenceModel
from chexpert.utils.wrappers import Wrapper

from flavor.serve.apps import InferAPP
from flavor.serve.models import InferClassificationOutput, ModelOutput
from flavor.serve.strategies import (
    AiCOCOClassificationOutputStrategy,
    AiCOCOInputStrategy,
)


class ClassificationInferenceModel(BaseInferenceModel):
    def __init__(self, output_data_model: InferClassificationOutput):
        super().__init__(output_data_model=output_data_model)

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

    def define_inference_network(self):
        return Wrapper("chexpert/instances/optimized_model.h5")

    def define_categories(self):
        categories = {
            0: {"name": "Atelectasis"},
            1: {"name": "Cardiomegaly"},
            2: {"name": "Consolidation"},
            3: {"name": "Edema"},
            4: {"name": "Enlarged Cardiomediastinum"},
            5: {"name": "Fracture"},
            6: {"name": "Lung Lesion"},
            7: {"name": "Lung Opacity"},
            8: {"name": "No Finding"},
            9: {"name": "Pleural Effusion"},
            10: {"name": "Pleural Other"},
            11: {"name": "Pneumonia"},
            12: {"name": "Pneumothorax"},
            13: {"name": "Support Devices"},
        }
        return categories

    def define_regressions(self):
        return None

    def preprocess(self, data_filenames: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        img = cv2.imread(data_filenames[0], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32)

        return img, data_filenames

    def postprocess(self, model_out: np.ndarray) -> np.ndarray:
        format_output = np.zeros(len(self.categories))
        for k, v in self.categories.items():
            category = v["name"]
            format_output[k] = int(model_out[category] > self.thesholds[category])

        return format_output

    def inference(self, data_filenames: Sequence[str]) -> Tuple[ModelOutput, List[str]]:
        data, sorted_data_filenames = self.preprocess(data_filenames)
        model_out = self.network.predict(data)
        model_out = self.postprocess(model_out)

        return model_out, sorted_data_filenames


if __name__ == "__main__":

    app = InferAPP(
        infer_function=ClassificationInferenceModel(output_data_model=InferClassificationOutput),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCOClassificationOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9999)))
