import torch

from .base_aicoco_inference_model import BaseAiCOCOInferenceModel


class GradioInferenceModel(BaseAiCOCOInferenceModel):
    def __call__(self, **net_input):
        categories = self.set_categories()
        regressions = self.set_regressions()

        data, _, metadata = self.data_reader(**net_input)

        with torch.no_grad():
            x = self.preprocess(data)
            out = self.inference(x)
            out = self.postprocess(out, metadata=metadata)
        result = self.output_formatter(
            out, categories=categories, regressions=regressions, data=x, **net_input
        )

        return result
