import os
from typing import Any, Dict, List

import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer

from flavor.serve.apps import InferAPP
from flavor.serve.strategies import (
    AiCOCOInputStrategy,
    AiCOCOSegmentationOutputStrategy,
)


def read_multiple_dicom(filenames):
    def sort_images_by_z_axis(filenames):

        images = []
        for fname in filenames:
            dicom_reader = sitk.ImageFileReader()
            dicom_reader.SetFileName(fname)
            dicom_reader.ReadImageInformation()

            images.append([dicom_reader, fname])

        zs = [float(dr.GetMetaData(key="0020|0032").split("\\")[-1]) for dr, _ in images]

        sort_inds = np.argsort(zs)[::-1]
        images = [images[s] for s in sort_inds]

        return images

    images = sort_images_by_z_axis(filenames)

    drs, fnames = zip(*images)
    fnames = list(fnames)

    simages = [sitk.GetArrayFromImage(dr.Execute()).squeeze() for dr in drs]
    volume = np.stack(simages)

    volume = np.expand_dims(volume, axis=0)

    return volume, fnames


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

        batch, data_filenames = read_multiple_dicom(data_filenames_l)

        model_out = self.inferer.apply(batch.squeeze(0))

        model_out = [
            np.expand_dims((model_out == i).astype(np.uint8), axis=0)
            for i in range(len(self.categories))
        ]
        model_out = np.concatenate(model_out, axis=0)

        return model_out, data_filenames

    def __call__(self, **kwargs):

        model_out, sorted_data_filenames = self.infer(**kwargs)
        result = self.make_infer_result(model_out, sorted_data_filenames, **kwargs)

        return result


if __name__ == "__main__":

    app = InferAPP(
        infer_function=Inferer(),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCOSegmentationOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9999)))
