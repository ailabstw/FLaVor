import os
from typing import Any, Dict, List

import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from flavor.serve.apps import InferAPP
from flavor.serve.strategies import AiCOCOInputStrategy, AiCOCORegressionOutputStrategy


def read_img(filename, **kwargs):

    img = Image.open(filename)

    transforms = ResNet18_Weights.DEFAULT.transforms()
    img = transforms(img).unsqueeze(0)

    return img


class Inferer:
    def __init__(self):

        self.regressions = {
            0: {"name": "reg0"},
            1: {"name": "reg1"},
        }

        self.inferer = resnet18(ResNet18_Weights.DEFAULT)
        self.inferer.fc = nn.Linear(512, 2)
        self.inferer = self.inferer.eval()
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
            "regressions": self.regressions,
            "model_out": model_out,
        }

    def infer(self, **kwargs) -> Dict[str, Any]:

        data_filenames_l = []
        for elem in kwargs["images"]:
            data_filenames_l.append(elem["physical_file_name"])

        img = read_img(data_filenames_l[0])

        model_out = self.inferer(img).squeeze(0)

        return model_out, data_filenames_l

    def __call__(self, **kwargs):

        model_out, sorted_data_filenames = self.infer(**kwargs)
        result = self.make_infer_result(model_out, sorted_data_filenames, **kwargs)

        return result


if __name__ == "__main__":

    app = InferAPP(
        infer_function=Inferer(),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCORegressionOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9999)))
