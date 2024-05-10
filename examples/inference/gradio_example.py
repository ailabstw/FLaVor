import os
from typing import Any, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer

from flavor.serve.apps import GradioInferAPP
from flavor.serve.inference import GradioInferenceModel
from flavor.serve.models import AiImage, InferCategory
from flavor.serve.strategies import GradioSegmentationStrategy


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

    def data_reader(
        self, files: Sequence[str], **kwargs
    ) -> Tuple[np.ndarray, List[str], Tuple[int]]:
        def sort_images_by_z_axis(filenames):

            sorted_reader_filename_pairs = []

            for f in filenames:
                dicom_reader = sitk.ImageFileReader()
                dicom_reader.SetFileName(f)
                dicom_reader.ReadImageInformation()

                sorted_reader_filename_pairs.append((dicom_reader, f))

            zs = [
                float(r.GetMetaData(key="0020|0032").split("\\")[-1])
                for r, _ in sorted_reader_filename_pairs
            ]

            sort_inds = np.argsort(zs)[::-1]
            sorted_reader_filename_pairs = [sorted_reader_filename_pairs[s] for s in sort_inds]

            return sorted_reader_filename_pairs

        pairs = sort_images_by_z_axis(files)

        readers, sorted_filenames = zip(*pairs)
        sorted_filenames = list(sorted_filenames)

        simages = [sitk.GetArrayFromImage(r.Execute()).squeeze() for r in readers]
        volume = np.stack(simages)
        volume = np.expand_dims(volume, axis=0)

        return volume, sorted_filenames, volume.shape[1:]

    def inference(self, x: np.ndarray) -> np.ndarray:
        return self.network.apply(np.squeeze(x, axis=0))

    def postprocess(self, out: Any, metadata: Any = None) -> Any:
        # (1, h, w) -> (6, h, w)
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
        data: Any,
        **kwargs
    ) -> Any:
        output = {"model_out": model_out, "images": images, "categories": categories, "data": data}
        return output


app = GradioInferAPP(
    infer_function=SegmentationGradioInferenceModel(),
    output_strategy=GradioSegmentationStrategy,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9000)))
