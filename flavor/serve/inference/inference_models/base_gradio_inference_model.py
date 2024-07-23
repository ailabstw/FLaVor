from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base_aicoco_inference_model import BaseAiCOCOInferenceModel


class GradioInferenceModel(BaseAiCOCOInferenceModel):
    """
    A model for performing inference on images using Gradio.

    This class inherits from `BaseAiCOCOImageInferenceModel` and implements the `__call__` method
    to perform the complete inference pipeline, especially for `output_formatter` taking extra argument `x`.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def define_inference_network(self) -> Callable:
        """
        Abstract method to define the inference network.

        Returns:
            Callable: The defined inference network instance.
                The return value would be assigned to `self.network`.
        """
        raise NotImplementedError

    @abstractmethod
    def set_categories(self) -> Optional[List[Dict[str, Any]]]:
        """
        Abstract method to set inference categories. Return `None` if no categories.

        Returns:
            List[Dict[str, Any]]: A list defining inference categories.
        """
        raise NotImplementedError

    @abstractmethod
    def set_regressions(self) -> Optional[List[Dict[str, Any]]]:
        """
        Abstract method to set inference regressions. Return `None` if no regressions.

        Returns:
            List[Dict[str, Any]]: A list defining inference regressions.
        """
        raise NotImplementedError

    @abstractmethod
    def data_reader(
        self, files: Optional[Sequence[str]] = None, **kwargs
    ) -> Tuple[Any, Optional[List[str]], Optional[Any]]:
        """
        Abstract method to read data for inference model.
        This method should return three things:
        1. data: in np.ndarray or torch.Tensor for inference model.
        2. modified_filenames: modified list of filenames if the order of `files` is altered (e.g., 3D multiple slices input).
        3. metadata: necessary metadata for the post-processing.

        Args:
            files (Sequence[str]): List of input filenames.

        Returns:
            Tuple[Any, Optional[List[str]], Optional[Any]]: A tuple containing data, modified filenames, and metadata.
        """
        raise NotImplementedError

    def preprocess(self, data: Any) -> Any:
        """
        A default operation for transformations which is identical transformation.

        Override it if you need other transformations like resizing or cropping, etc.

        Args:
            data (Any): Input data.

        Returns:
            Any: Preprocessed data.
        """
        return data

    def inference(self, x: Any) -> Any:
        """
        A default inference operation which performs forward operation of your defined network.

        Override it if needed.

        Args:
            x (Any): Input data.

        Returns:
            Any: Inference result.
        """

        return self.network(x)

    def postprocess(self, out: Any, metadata: Optional[Any] = None) -> Any:
        """
        A default operation for post-processing which is identical transformation.

        Override it if you need activations like softmax or sigmoid generating the prediction.

        Args:
            out (Any): Inference result.
            metadata (Any, optional): Additional metadata. Default: None.

        Returns:
            Any: Post-processed result.
        """

        return out

    def output_formatter(
        self,
        model_out: np.ndarray,
        categories: Optional[Sequence[Dict[str, Any]]] = None,
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        output = {
            "model_out": model_out,
            "categories": categories,
            "regressions": regressions,
        }
        output.update(kwargs)
        return output

    def __call__(self, *args, **kwargs):
        data, _, metadata = self.data_reader(**kwargs)

        x = self.preprocess(data)
        out = self.inference(x)
        out = self.postprocess(out, metadata=metadata)

        result = self.output_formatter(
            model_out=out,
            categories=self.categories,
            regressions=self.regressions,
            data=data,
        )

        return result
