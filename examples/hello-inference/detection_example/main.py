import os
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

from flavor.serve.apps import InferAPP
from flavor.serve.strategies import AiCOCODetectionOutputStrategy, AiCOCOInputStrategy


def read_img(filename, **kwargs):

    image = cv2.imread(filename)
    image = image.astype(np.float32)

    return image


class Inferer:
    def __init__(self):

        self.categories = {
            0: {"name": "RBC", "display": True},
            1: {"name": "WBC", "display": True},
            2: {"name": "Platelets", "display": True},
        }

        self.inferer = YOLO("best.pt")
        print("Loaded successfully.")

    def make_infer_result(
        self, model_out: List[List[float]], sorted_data_filenames: List[str], **kwargs
    ) -> Dict:

        images_path_table = {}
        for image in kwargs["images"]:
            images_path_table[image["physical_file_name"]] = image

        sort_images = [images_path_table[filename] for filename in sorted_data_filenames]

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

        return {
            "sorted_images": sort_images,
            "categories": self.categories,
            "model_out": format_output,
        }

    def infer(self, **kwargs) -> Dict[str, Any]:

        data_filenames_l = []
        for elem in kwargs["images"]:
            data_filenames_l.append(elem["physical_file_name"])

        img = read_img(data_filenames_l[0])

        model_out = self.inferer.predict(img, conf=0.7)[0]

        return model_out, data_filenames_l

    def __call__(self, **kwargs):

        model_out, sorted_data_filenames = self.infer(**kwargs)
        result = self.make_infer_result(model_out, sorted_data_filenames, **kwargs)

        return result


if __name__ == "__main__":

    app = InferAPP(
        infer_function=Inferer(),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCODetectionOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9000)))
