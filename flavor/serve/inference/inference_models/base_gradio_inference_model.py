from .base_aicoco_inference_model import BaseAiCOCOImageInferenceModel


class GradioInferenceModel(BaseAiCOCOImageInferenceModel):
    """
    A model for performing inference on images using Gradio.

    This class inherits from `BaseAiCOCOImageInferenceModel` and implements the `__call__` method
    to perform the complete inference pipeline, especially for `output_formatter` taking extra argument `x`.
    """

    def __call__(self, **inputs: dict):
        self.categories = self.set_categories()
        self.regressions = self.set_regressions()

        data, _, metadata = self.data_reader(**inputs)

        x = self.preprocess(data)
        out = self.inference(x)
        out = self.postprocess(out, metadata=metadata)
        result = self.output_formatter(
            out, categories=self.categories, regressions=self.regressions, data=x, **inputs
        )

        return result
