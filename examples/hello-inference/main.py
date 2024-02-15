import os
from typing import Dict, List

import numpy as np
from lungmask import LMInferer
from reader import read_multiple_dicom

from flavor.serve.apps import InferAPP
from flavor.serve.strategies import (
    AiCOCOInputStrategy,
    AiCOCOSegmentationOutputStrategy,
)


class Inferer:
    def __init__(self):

        self.categories = {
            0: {"name": "Background", "display": False},
            1: {"name": "Left Upper Lobe", "display": True},
            2: {"name": "Left Lower Lobe", "display": True},
            3: {"name": "Right Upper Lobe", "display": True},
            4: {"name": "Right Middle Lobe", "display": True},
            5: {"name": "Right Lower Lobe", "display": True},
        }

        self.inferer = LMInferer(modelname="LTRCLobes", fillmodel="R231")
        print("Loaded successfully.")

    def make_infer_res(self, model_out: np.ndarray, sorted_data_filenames: List, **kwargs) -> Dict:

        images_path_table = {}
        for image in kwargs["images"]:
            images_path_table[image["physical_file_name"]] = image

        sort_images = [images_path_table[filename] for filename in sorted_data_filenames]

        return {
            "sorted_images": sort_images,
            "categories": self.categories,
            "model_out": model_out,
        }

    def infer(self, **kwargs) -> Dict:

        data_filenames = []
        for elem in kwargs["images"]:
            data_filenames.append(elem["physical_file_name"])

        batch = read_multiple_dicom(data_filenames)

        model_out = self.inferer.apply(batch["data"].squeeze(0))

        model_out = [
            np.expand_dims((model_out == i).astype(np.uint8), axis=0)
            for i in range(len(self.categories))
        ]
        model_out = np.concatenate(model_out, axis=0)

        return model_out, batch["data_filenames"]

    def __call__(self, **kwargs):

        batch_out, sorted_data_filenames = self.infer(**kwargs)
        result = self.make_infer_res(batch_out, sorted_data_filenames, **kwargs)

        return result


if __name__ == "__main__":

    app = InferAPP(
        infer_function=Inferer(),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCOSegmentationOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9000)))
