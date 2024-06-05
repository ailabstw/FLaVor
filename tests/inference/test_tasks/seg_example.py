import os
from typing import Any, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer

from flavor.serve.apps import InferAPP
from flavor.serve.inference import (
    BaseAiCOCOImageInferenceModel,
    BaseAiCOCOImageInputDataModel,
    BaseAiCOCOImageOutputDataModel,
)
from flavor.serve.models import AiImage, InferCategory
from flavor.serve.strategies import AiCOCOSegmentationOutputStrategy


class SegmentationInferenceModel(BaseAiCOCOImageInferenceModel):
    def __init__(self):
        self.formatter = AiCOCOSegmentationOutputStrategy()
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

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[np.ndarray, None, None]:
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

    def postprocess(self, out: Any, metadata: Any = None) -> Any:
        # (1, h, w) -> (c, h, w)
        out = [
            np.expand_dims((out == i).astype(np.uint8), axis=0)
            for i in range(6)  # or len(self.categories)
        ]
        out = np.concatenate(out, axis=0)
        return out

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        categories: Sequence[InferCategory],
        **kwargs
    ) -> Any:

        output = self.formatter(model_out=model_out, images=images, categories=categories)
        return output


app = InferAPP(
    infer_function=SegmentationInferenceModel(),
    input_data_model=BaseAiCOCOImageInputDataModel,
    output_data_model=BaseAiCOCOImageOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
