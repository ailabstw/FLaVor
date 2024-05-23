from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from flavor.serve.apps import InferAPP
from flavor.serve.inference import (
    BaseAiCOCOInferenceModel,
    BaseAiCOCOInputDataModel,
    BaseAiCOCOOutputDataModel,
)
from flavor.serve.models import AiImage, InferCategory, NpArray
from flavor.serve.strategies import AiCOCOSegmentationOutputStrategy

from .sam_triton_inference_model import SamDecoderTritonInferenceModel


class SamAiCOCODecoderInferenceModel(BaseAiCOCOInferenceModel):
    def __init__(
        self,
        triton_url="triton:8000",
        triton_network_name="sam_decoder",
    ):
        """
        Segment-anything decoder.

        Arguments
            output_data_model = None
            triton_url: str = "triton:8000"
                url to connect to triton inference server
                for detail, see kubernetes Service DNS
            triton_network_name: str = "sam_decoder"
                network name of SAM decoder registered in triton inference server

        Request input (form data):
            image_embeddings: json dumped NpArray
                image_embeddings returned from SamEncoder.
                shape should be (1, 256, 64, 64)
            point_coords: list[list[list[int]]]
                point of interest that should be segmented
                shape is (1, num_points, 2)
            orig_im_size: list[int]
                original image shape in order of (h, w)
            images: list[AiImage]
                list of images in type AiImage
            mask_input: json dumped NpArray
                `low_res_masks` (logits) returned from SamDecoder
                shape should be (1, 1, 256, 256)

        Return
            sorted_images
            categories
            regressions
            model_out: np.ndarray
                resized mask in binary form
            (the four above are for AiCOCOSegmentationOutputStrategy)
            mask_logits: np.ndarray
                raw mask logits
        """
        self.triton_url = triton_url
        self.triton_network_name = triton_network_name
        self.formatter = AiCOCOSegmentationOutputStrategy()
        super().__init__()

    def set_categories(self):
        categories = [
            {"name": "Foreground", "display": True},
        ]
        return categories

    def set_regressions(self):
        return None

    def define_inference_network(self):
        decoder = SamDecoderTritonInferenceModel(
            triton_url=self.triton_url, model_name=self.triton_network_name, is_shared_memory=False
        )
        return decoder

    def data_reader(
        self, image_embeddings, point_coords, orig_im_size, mask_input=None, **kwargs
    ) -> Tuple[Any, Optional[List[str]], Optional[Any]]:
        """
        - `point_labels` and `has_mask_input` are automatically set
        - all `point_label` are set to 1. the inference model does not accept `point_label=0`

        - `has_mask_input` is set to [1] when `mask_input` is provided
        - `point_labels` is set to [[1, ...]] depend on the number of points in `point_coords`
        """
        # image_embeddings = json.loads(image_embeddings)

        point_labels = [[1 for _ in range(len(point_coords[0]))]]

        data_dict = {
            "image_embeddings": image_embeddings["array"],
            "point_coords": np.array(point_coords),
            "point_labels": np.array(point_labels),
            "orig_im_size": np.array(orig_im_size),
        }
        if mask_input is not None:

            data_dict["mask_input"] = mask_input["array"]
            data_dict["has_mask_input"] = np.array([1])

        return data_dict, None, None

    def preprocess(self, data_dict: Any):
        return data_dict

    def inference(self, data_dict):
        results = self.network.predict(data_dict)
        return results

    def postprocess(self, results: Any, metadata: Any = None) -> Any:
        results["mask_bin"] = (results["masks"] > 0).astype(np.uint8)
        return results

    def output_formatter(
        self,
        out: Any,
        images: Optional[Sequence[AiImage]],
        categories: Sequence[InferCategory],
        **kwargs
    ) -> Any:
        mask_bin = out["mask_bin"]
        mask_logits = out["low_res_masks"]

        aicoco_output = self.formatter(model_out=mask_bin, images=images, categories=categories)
        infer_output = {
            **aicoco_output,
            "mask_bin": mask_bin,
            "mask_logits": mask_logits,
        }

        return infer_output


class InputDataModel(BaseAiCOCOInputDataModel):
    image_embeddings: NpArray
    point_coords: Sequence[Sequence[Sequence[int]]]
    orig_im_size: Sequence[int]
    mask_input: Optional[NpArray] = None


class OutputDataModel(BaseAiCOCOOutputDataModel):
    mask_bin: NpArray
    mask_logits: NpArray


sam_decoder_app = InferAPP(
    infer_function=SamAiCOCODecoderInferenceModel(
        triton_url="triton:8000",
        triton_network_name="sam_decoder",
    ),
    input_data_model=InputDataModel,
    output_data_model=OutputDataModel,
)

if __name__ == "__main__":
    sam_decoder_app.run(port=9111)
