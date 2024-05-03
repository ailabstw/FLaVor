import os
from typing import Any, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer

from flavor.serve.apps import GradioInferAPP
from flavor.serve.inference import GradioInferenceModel
from flavor.serve.models import AiImage, InferCategory
from flavor.serve.strategies import GradioInputStrategy, GradioSegmentationStrategy


class SegmentationGradioInferenceModel(GradioInferenceModel):
    def __init__(SetFileName):
        super().__init__()

    def define_inference_network(self):
        return LMInferer(modelname="LTRCLobes", fillmodel="R231")

    def set_categories(self):
        categories = [
            {"name": "Background", "display": False},
            {"name": "Left Upper Lobe", "display": True},
            {"name": "Left Lower Lobe", "display": True},
            {"name": "Right Upper Lobe", "display": True},
            {"name": "Right Middle Lobe", "display": True},
            {"name": "Right Lower Lobe", "display": True},
        ]
        return categories

    def set_regressions(self):
        return None

    def data_reader(self, images: Sequence[AiImage], **kwargs) -> Tuple[np.ndarray, None, None]:
        files = []
        for image in images:
            files.append(image.pop("physical_file_name"))
        dicom_reader = sitk.ImageFileReader()
        dicom_reader.SetFileName(files[0])
        dicom_reader.ReadImageInformation()
        dicom = sitk.GetArrayFromImage(dicom_reader.Execute()).squeeze()

        return dicom, None, None

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        data = np.expand_dims(data, axis=0)
        return data

    def inference(self, x: np.ndarray) -> np.ndarray:
        return self.network.apply(x)

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        categories: Sequence[InferCategory],
        data: Any,
        **kwargs
    ) -> Any:

        # (1, h, w) -> (c, h, w)
        model_out = [
            np.expand_dims((model_out == i).astype(np.uint8), axis=0)
            for i in range(len(categories))
        ]
        model_out = np.concatenate(model_out, axis=0)
        output = {"model_out": model_out, "images": images, "categories": categories, "data": data}
        return output


app = GradioInferAPP(
    infer_function=SegmentationGradioInferenceModel(),
    input_strategy=GradioInputStrategy,
    output_strategy=GradioSegmentationStrategy,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9000)))
