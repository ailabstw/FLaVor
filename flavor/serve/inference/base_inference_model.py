import abc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import TypeAdapter

from flavor.serve.models import (
    InferCategory,
    InferInput,
    InferInputImage,
    InferOutput,
    InferRegression,
    ModelOutput,
)


class BaseInferenceModel:
    """
    Base class for defining inference models.

    This class serves as a template for implementing inference functionality for various machine learning or deep learning models.
    Subclasses must override abstract methods to define model-specific behavior.

    Attributes:
        output_data_model (InferOutput): The output data model expected from the inference process.
        network (Callable): The inference network or model.
        categories (Dict[int, InferCategory]): Dictionary defining inference categories.
        regressions (Dict[int, InferRegression]): Dictionary defining inference regressions.
    """

    def __init__(self, output_data_model: InferOutput):
        self.output_data_model = output_data_model
        self.network = self.define_inference_network()
        self.categories = self.define_categories()
        self.regressions = self.define_regressions()

    @abc.abstractmethod
    def define_inference_network(self) -> Callable:
        """
        Abstract method to define the inference network.

        Returns:
            Callable: The defined inference network instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def define_categories(self) -> Dict[int, InferCategory]:
        """
        Abstract method to define inference categories. Return `None` if no categories.

        Returns:
            Dict[int, InferCategory]: Dictionary defining inference categories.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def define_regressions(self) -> Dict[int, InferRegression]:
        """
        Abstract method to define inference regressions. Return `None` if no regressions.

        Returns:
            Dict[int, InferRegression]: Dictionary defining inference regressions.
        """
        raise NotImplementedError

    def make_infer_result(
        self,
        model_out: ModelOutput,
        sorted_data_filenames: Optional[Sequence[str]] = None,
        **input_aicoco
    ) -> InferOutput:
        """
        Formulates inference output and validates output type based on `output_data_model`.
        If `sorted_data_filenames` is provided, return the order of `image` field will based on it.

        Args:
            model_out (ModelOutput): Inference model output.
            sorted_data_filenames (Optional[Sequence[str]]):
                List of data filenames arranged in order if input data is in 3D. Default: None.
            **input_aicoco: Additional input data.

        Returns:
            InferOutput: Formulated inference model output.
        """

        if sorted_data_filenames is not None:
            images_path_table = {}
            for image in input_aicoco["images"]:
                images_path_table[image["physical_file_name"]] = image

            images = [images_path_table[filename] for filename in sorted_data_filenames]

        else:
            images = sorted(input_aicoco["images"], key=lambda x: x["index"])

        infer_output = {
            "images": images,
            "categories": self.categories,
            "regressions": self.regressions,
            "model_out": model_out,
        }

        self.output_data_model.model_validate(infer_output)

        return infer_output

    def get_data_filename(self, images: Sequence[InferInputImage]) -> List[str]:
        """
        Retrieves the physical location of input data in the local machine.
        Only override it when it is necessary.

        Args:
            images (Sequence[InferInputImage]): Input images with associated metadata.

        Returns:
            List[str]: Physical filenames of input data.
        """
        ta = TypeAdapter(Sequence[InferInputImage])
        ta.validate_python(images)

        data_filenames = []
        for elem in images:
            data_filenames.append(elem["physical_file_name"])

        return data_filenames

    @abc.abstractmethod
    def preprocess(self, data_filenames: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Abstract method for data preprocessing.
        The `preprocess` method is responsible for preparing the input data for the inference process.
        This typically involves reading the data from disk, performing any necessary transformations
        or preprocessing steps, and organizing it into a format that can be consumed by the inference model.

        Note that for this method should also return sorted filename based on its slice or index number if input data is in 3D.
        This is because for dicom images you would only know their order after reading them.
        """
        raise NotImplementedError

    def postprocess(self, model_out: Any) -> ModelOutput:
        """Handles additional postprocessing of model output."""
        return model_out

    def __call__(self, **input_aicoco: InferInput) -> InferOutput:
        """Runs the inference model.

        Returns:
            InferOutput: Inference model output for output strategy.

        We provide an implementation template for inference model. You should customize properly for your application.
        Three operations are involved in the following example.
        1. Input data filename parser: Parse the data filename for further data reading.
        2. Inference process: Perform inference process, typically involving the following steps:
            a. Preprocessing: The input data is preprocessed to prepare it for feeding into the inference model.
                This may include loading data from disk, performing transformations, and formatting it appropriately.
            b. Model forward: The preprocessed data is passed through the inference model, which produces predictions or outputs based on the input.
            c. Postprocessing: Optionally, the raw output from the model may undergo additional postprocessing steps to refine or format the results as needed.
        3. Inference model output formatter: Formulate the inference model output in a compatible format for output strategy.
            See `InferOutput`.
        """

        # input data filename parser
        data_filenames = self.get_data_filename(**input_aicoco)

        # inference
        data = self.preprocess(data_filenames)
        model_out = self.network(data)
        model_out = self.postprocess(model_out)

        # inference model output formatter
        result = self.make_infer_result(model_out, **input_aicoco)

        return result
