from typing import Optional, Sequence

import numpy as np

from flavor.serve.models import InferOutput, ModelOut

from .base_aicoco_inference_model import BaseAiCOCOInferenceModel


class GradioInferenceModel(BaseAiCOCOInferenceModel):
    def make_infer_result(
        self,
        model_out: ModelOut,
        data: np.ndarray,
        sorted_data_filenames: Optional[Sequence[str]] = None,
        **infer_input
    ) -> InferOutput:
        infer_output = super().make_infer_result(model_out, sorted_data_filenames, **infer_input)
        infer_output["data"] = data

        return infer_output

    def __call__(self, **infer_input) -> InferOutput:
        # input data filename parser
        data_filenames = self.get_data_filename(**infer_input)

        # inference
        data = self.preprocess(data_filenames)
        model_out = self.network(data)
        model_out = self.postprocess(model_out)

        # inference model output formatter
        result = self.make_infer_result(model_out, data=data, **infer_input)

        return result
