from typing import List, Sequence, Tuple

import numpy as np
from base_inference_model import BaseInferenceModel

from flavor.serve.models import InferOutput, ModelOutput


class GradioInferenceModel(BaseInferenceModel):
    def make_infer_result(
        self,
        model_out: ModelOutput,
        sorted_data_filenames: Sequence[str],
        data: np.ndarray,
        **input_aicoco
    ) -> InferOutput:
        infer_output = super().make_infer_result(model_out, sorted_data_filenames, **input_aicoco)
        infer_output["data"] = data

        return infer_output

    def inference(self, data_filenames: Sequence[str]) -> Tuple[ModelOutput, List[str]]:

        data, sorted_data_filenames = self.preprocess(data_filenames)
        model_out = self.network(data)
        model_out = self.postprocess(model_out)

        return model_out, sorted_data_filenames, data

    def __call__(self, **input_aicoco) -> InferOutput:
        data_filenames = self.get_data_filename(**input_aicoco)
        model_out, sorted_data_filenames, data = self.inference(data_filenames)
        result = self.make_infer_result(model_out, sorted_data_filenames, data, **input_aicoco)

        return result
