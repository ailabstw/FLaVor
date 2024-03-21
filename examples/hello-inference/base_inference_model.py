from typing import Any, Callable, List, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, TypeAdapter

from flavor.serve.models import (
    InferCategories,
    InferInputImage,
    InferOutput,
    InferRegressions,
    ModelOutput,
)


class BaseInferenceModel:
    def __init__(self, output_data_model: BaseModel):
        self.output_data_model = output_data_model
        self.network = self.define_inference_network()
        self.categories = self.define_categories()
        self.regressions = self.define_regressions()

    def define_inference_network(self) -> Callable:
        raise NotImplementedError

    def define_categories(self) -> InferCategories:
        raise NotImplementedError

    def define_regressions(self) -> InferRegressions:
        raise NotImplementedError

    def make_infer_result(
        self, model_out: ModelOutput, sorted_data_filenames: Sequence[str], **input_aicoco
    ) -> InferOutput:

        images_path_table = {}
        for image in input_aicoco["images"]:
            images_path_table[image["physical_file_name"]] = image

        sort_images_field = [images_path_table[filename] for filename in sorted_data_filenames]

        infer_output = {
            "sorted_images": sort_images_field,
            "categories": self.categories,
            "regressions": self.regressions,
            "model_out": model_out,
        }

        ta = TypeAdapter(self.output_data_model)
        ta.validate_python(infer_output)

        return infer_output

    def get_data_filename(self, images: Sequence[InferInputImage]) -> List[str]:

        ta = TypeAdapter(Sequence[InferInputImage])
        ta.validate_python(images)

        data_filenames = []
        for elem in images:
            data_filenames.append(elem["physical_file_name"])

        return data_filenames

    def preprocess(self, data_filenames: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        """data reading and other data transformations.
        Since input to the inference model would be a dictionary, in order to retrieve input data, data reading has to be handled specifically.
        """
        raise NotImplementedError

    def postprocess(self, model_out: Any) -> ModelOutput:

        return model_out

    def inference(self, data_filenames: Sequence[str]) -> Tuple[InferOutput, List[str]]:
        """model inference function

        recommended steps:
            1. read data from `data_filenames` (sort based on indices or slice number if multiple elements found)
            2. run model inference
            3. pack `model_out` in a specific format based on the task

        Args:
            data_filenames_l (Sequence[str]): list of the directory of input files in local machine

        Returns:
            Tuple[InferOutput,  List[str]]: `model_out` and `sorted_data_filenames`
        """

        data, sorted_data_filenames = self.preprocess(data_filenames)
        model_out = self.network(data)
        model_out = self.postprocess(model_out)

        return model_out, sorted_data_filenames

    def __call__(self, **input_aicoco: InferInputImage) -> InferOutput:

        data_filenames = self.get_data_filename(**input_aicoco)
        model_out, sorted_data_filenames = self.inference(data_filenames)
        result = self.make_infer_result(model_out, sorted_data_filenames, **input_aicoco)

        return result
