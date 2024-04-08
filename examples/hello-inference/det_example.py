import os
from typing import Any, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from flavor.serve.apps import InferAPP
from flavor.serve.inference import BaseInferenceModel
from flavor.serve.models import DetModelOut, InferDetectionOutput, InferInput
from flavor.serve.strategies import AiCOCODetectionOutputStrategy, AiCOCOInputStrategy


class DetectionInferenceModel(BaseInferenceModel):
    def __init__(self, output_data_model: InferDetectionOutput):
        super().__init__(output_data_model=output_data_model)

    def define_inference_network(self):
        return YOLO("best.pt")

    def define_categories(self):
        categories = {
            0: {"name": "RBC", "display": True},
            1: {"name": "WBC", "display": True},
            2: {"name": "Platelets", "display": True},
        }
        return categories

    def define_regressions(self):
        return None

    def preprocess(self, data_filenames: Sequence[str]) -> np.ndarray:
        image = cv2.imread(data_filenames[0])
        image = image.astype(np.float32)

        return image

    def postprocess(self, model_out: Any) -> DetModelOut:

        format_output = {
            "bbox_pred": [],
            "cls_pred": [],
            "confidence_score": [],
        }

        for obj in model_out.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = obj
            format_output["bbox_pred"].append([int(x1), int(y1), int(x2), int(y2)])
            cls_pred = np.zeros(len(self.categories))
            cls_pred[int(class_id)] = 1
            format_output["cls_pred"].append(cls_pred)
            format_output["confidence_score"].append(score)

        return format_output

    def __call__(self, **infer_input: InferInput) -> InferDetectionOutput:
        # input data filename parser
        data_filenames = self.get_data_filename(**infer_input)

        # inference
        data = self.preprocess(data_filenames)
        model_out = self.network.predict(data, conf=0.7)[0]
        model_out = self.postprocess(model_out)

        # inference model output formatter
        result = self.make_infer_result(model_out, **infer_input)

        return result


if __name__ == "__main__":

    app = InferAPP(
        infer_function=DetectionInferenceModel(output_data_model=InferDetectionOutput),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCODetectionOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9000)))
