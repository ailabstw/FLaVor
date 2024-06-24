from .base_aicoco_inference_model import BaseAiCOCOImageInferenceModel


class GradioInferenceModel(BaseAiCOCOImageInferenceModel):
    def __call__(self, **net_input):
        self.categories = self.set_categories()
        self.regressions = self.set_regressions()

        data, _, metadata = self.data_reader(**net_input)

        x = self.preprocess(data)
        out = self.inference(x)
        out = self.postprocess(out, metadata=metadata)
        result = self.output_formatter(
            out, categories=self.categories, regressions=self.regressions, data=x, **net_input
        )

        return result
