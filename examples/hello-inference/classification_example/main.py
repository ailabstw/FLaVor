import os
from typing import Any, Dict, List

import cv2
import numpy as np
from chexpert.utils.wrappers import Wrapper

from flavor.serve.apps import InferAPP
from flavor.serve.strategies import (
    AiCOCOClassificationOutputStrategy,
    AiCOCOInputStrategy,
)


def read_img(filename, **kwargs):

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    image = img.astype(np.float32)

    return image


class Inferer:
    def __init__(self):

        self.categories = {
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

        self.inferer = Wrapper("chexpert/instances/optimized_model.h5")
        print("Loaded successfully.")

    def make_infer_result(
        self, model_out: np.ndarray, sorted_data_filenames: List[str], **kwargs
    ) -> Dict:

        images_path_table = {}
        for image in kwargs["images"]:
            images_path_table[image["physical_file_name"]] = image

        sort_images = [images_path_table[filename] for filename in sorted_data_filenames]

        return {
            "sorted_images": sort_images,
            "categories": self.categories,
            "model_out": model_out,
        }

    def infer(self, **kwargs) -> Dict[str, Any]:

        data_filenames_l = []
        for elem in kwargs["images"]:
            data_filenames_l.append(elem["physical_file_name"])

        img = read_img(data_filenames_l[0])

        model_out = self.inferer.predict(img)

        format_output = np.zeros(len(self.categories))
        for k, v in self.categories.items():
            category = v["name"]
            format_output[k] = int(model_out[category] > self.thesholds[category])

        return format_output, data_filenames_l

    def __call__(self, **kwargs):

        model_out, sorted_data_filenames = self.infer(**kwargs)
        result = self.make_infer_result(model_out, sorted_data_filenames, **kwargs)

        return result


if __name__ == "__main__":

    app = InferAPP(
        infer_function=Inferer(),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCOClassificationOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9000)))
