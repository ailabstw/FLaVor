import os
from typing import List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer

from flavor.serve.apps import InferAPP
from flavor.serve.inference import BaseInferenceModel
from flavor.serve.models import InferInput, InferSegmentationOutput, ModelOut
from flavor.serve.strategies import (
    AiCOCOInputStrategy,
    AiCOCOSegmentationOutputStrategy,
)


class SegmentationInferenceModel(BaseInferenceModel):
    def __init__(self, output_data_model: InferSegmentationOutput):
        super().__init__(output_data_model=output_data_model)

    def define_inference_network(self):
        return LMInferer(modelname="LTRCLobes", fillmodel="R231")

    def define_categories(self):
        categories = {
            0: {"name": "Background", "display": False},
            1: {"name": "Left Upper Lobe", "display": True},
            2: {"name": "Left Lower Lobe", "display": True},
            3: {"name": "Right Upper Lobe", "display": True},
            4: {"name": "Right Middle Lobe", "display": True},
            5: {"name": "Right Lower Lobe", "display": True},
        }
        return categories

    def define_regressions(self):
        return None

    def preprocess(self, data_filenames: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
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

        images = sort_images_by_z_axis(data_filenames)

        drs, fnames = zip(*images)
        fnames = list(fnames)

        simages = [sitk.GetArrayFromImage(dr.Execute()).squeeze() for dr in drs]
        volume = np.stack(simages)

        volume = np.expand_dims(volume, axis=0)

        return volume, fnames

    def postprocess(self, model_out: np.ndarray) -> ModelOut:

        model_out = [
            np.expand_dims((model_out == i).astype(np.uint8), axis=0)
            for i in range(len(self.categories))
        ]
        model_out = np.concatenate(model_out, axis=0)

        return model_out

    def __call__(self, **infer_input: InferInput) -> InferSegmentationOutput:
        # input data filename parser
        data_filenames = self.get_data_filename(**infer_input)

        # inference
        data, sorted_data_filenames = self.preprocess(data_filenames)
        model_out = self.network.apply(data.squeeze(0))
        model_out = self.postprocess(model_out)

        # inference model output formatter
        result = self.make_infer_result(model_out, sorted_data_filenames, **infer_input)

        return result


if __name__ == "__main__":

    app = InferAPP(
        infer_function=SegmentationInferenceModel(output_data_model=InferSegmentationOutput),
        input_strategy=AiCOCOInputStrategy,
        output_strategy=AiCOCOSegmentationOutputStrategy,
    )
    app.run(port=int(os.getenv("PORT", 9000)))
