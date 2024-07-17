from typing import Any, Dict, Optional, Sequence

import numpy as np

from .base_aicoco_inference_model import BaseAiCOCOImageInferenceModel


class GradioInferenceModel(BaseAiCOCOImageInferenceModel):
    """
    A model for performing inference on images using Gradio.

    This class inherits from `BaseAiCOCOImageInferenceModel` and implements the `__call__` method
    to perform the complete inference pipeline, especially for `output_formatter` taking extra argument `x`.
    """

    def __init__(self):
        super().__init__()

    def output_formatter(
        self,
        model_out: np.ndarray,
        data: np.ndarray,
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        output = {
            "model_out": model_out,
            "data": data,
            "categories": categories,
            "regressions": regressions,
        }
        return output

    def __call__(self, **inputs: Dict):

        data, _, metadata = self.data_reader(**inputs)

        x = self.preprocess(data)
        out = self.inference(x)
        out = self.postprocess(out, metadata=metadata)

        result = self.output_formatter(
            model_out=out, data=data, categories=self.categories, regressions=self.regressions
        )

        return result
