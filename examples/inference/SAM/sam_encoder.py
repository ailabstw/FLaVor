from typing import Any, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from flavor.serve.apps import InferAPP
from flavor.serve.inference import BaseAiCOCOInferenceModel, BaseAiCOCOInputDataModel
from flavor.serve.models import NpArray

from .sam_triton_inference_model import SamEncoderInferenceModel


class SamEncoderAiCOCOInferenceModel(BaseAiCOCOInferenceModel):
    def __init__(
        self,
        triton_url="triton:8000",
        triton_network_name="sam_encoder",
    ):
        """
        Segment-anything encoder.

        Arguments
            output_data_model = None
            triton_url: str = "triton:8000"
                url to connect to triton inference server
                for detail, see kubernetes Service DNS
            triton_network_name: str = "sam_encoder"
                network name of SAM encoder registered in triton inference server

        Request input (form data):
            files: formdata
            images: list[AiImage]
                list of images in type AiImage

        Return
            embeddings: NpArray
                jsonified image embeddings
            original_shapes: list[int]
                shape (h, w) of original image
        """
        self.triton_url = triton_url
        self.triton_network_name = triton_network_name
        super().__init__()

    def set_categories(self):
        return None

    def set_regressions(self):
        return None

    def define_inference_network(self):
        encoder = SamEncoderInferenceModel(
            triton_url=self.triton_url,
            model_name=self.triton_network_name,
            is_shared_memory=False,
        )
        return encoder

    def data_reader(
        self, files: Sequence[str], **kwargs
    ) -> Tuple[Any, Optional[List[str]], Optional[Any]]:
        # do nothing
        return files, None, None

    def preprocess(self, data: Any):
        return data

    def inference(self, x):
        results = self.network.predict(x)
        return results

    def output_formatter(self, model_out: Any, **kwargs) -> Any:
        embeddings = model_out["embeddings"]
        original_shapes = model_out["original_shapes"][0].tolist()
        output_dict = {
            "embeddings": embeddings,
            "original_shapes": original_shapes,
        }

        return output_dict


class OutputDataModel(BaseModel):
    embeddings: NpArray
    original_shapes: Sequence[int]


sam_encoder_app = InferAPP(
    infer_function=SamEncoderAiCOCOInferenceModel(
        triton_url="triton.user-hannchyun-chen:8000",
        triton_network_name="sam_encoder",
    ),
    input_data_model=BaseAiCOCOInputDataModel,
    output_data_model=OutputDataModel,
)

if __name__ == "__main__":
    sam_encoder_app.run(port=9111)
