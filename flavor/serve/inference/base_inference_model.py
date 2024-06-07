from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence

from flavor.serve.models import InferCategory, InferRegression


class BaseInferenceModel(ABC):
    @abstractmethod
    def __call__(self) -> Any:
        pass


class BaseAiCOCOInferenceModel(BaseInferenceModel):
    def __init__(self):
        self.network = self.define_inference_network()
        self.categories = self.set_categories()
        self.check_categories()

        self.regressions = self.set_regressions()
        self.check_regressions()

    def check_categories(self):
        """
        Check if defined categories field is valid for AiCOCO format.
        """
        if self.categories is not None:
            if isinstance(self.categories, Sequence):
                assert all(
                    InferCategory.model_validate(c) for c in self.categories
                ), "Not all elements in `self.categories` is valid for category structure."
            else:
                raise TypeError("`categories` should have type of `Sequence[InferCategory]`.")

    def check_regressions(self):
        """
        Check if defined regressions field is valid for AiCOCO format.
        """
        if self.regressions is not None:
            if isinstance(self.regressions, Sequence):
                assert all(
                    InferRegression.model_validate(c) for c in self.regressions
                ), "Not all elements in `self.regressions` is valid for regression structure."
            else:
                raise TypeError("`regressions` should have type of `Sequence[InferRegression]`.")

    @abstractmethod
    def define_inference_network(self) -> Callable:
        """
        Abstract method to define the inference network.

        Returns:
            Callable: The defined inference network instance.
                The return value would be assigned to `self.network`.
        """
        pass

    @abstractmethod
    def set_categories(self) -> Optional[List[InferCategory]]:
        """
        Abstract method to set inference categories. Return `None` if no categories.

        Returns:
            List[InferCategory]: A list defining inference categories.
        """
        pass

    @abstractmethod
    def set_regressions(self) -> Optional[List[InferRegression]]:
        """
        Abstract method to set inference regressions. Return `None` if no regressions.

        Returns:
            List[InferRegression]: A list defining inference regressions.
        """
        pass

    @abstractmethod
    def preprocess(self, net_input: Any) -> Any:
        pass
        """
        Abstract method to preprocess the input data where transformations like resizing and cropping operated.

        Args:
            data (Any): Input data.

        Returns:
            Any: Preprocessed data.
        """

    @abstractmethod
    def inference(self, x: Any) -> Any:
        pass
        """
        Abstract method to perform inference.

        Override it if needed.

        Args:
            x (Any): Input data.

        Returns:
            Any: Inference result.
        """

    @abstractmethod
    def postprocess(self, out: Any) -> Any:
        pass
        """
        Abstract method to post-process the inference result where activations like softmax or sigmoid performed.

        Args:
            out (Any): Inference result.

        Returns:
            Any: Post-processed result.
        """

    @abstractmethod
    def output_formatter(
        self,
        model_out: Any,
        categories: Optional[Sequence[InferCategory]] = None,
        regressions: Optional[Sequence[InferRegression]] = None,
        **kwargs,
    ) -> Any:
        pass
        """
        Abstract method to format the output of inference model.
        This is just a template for you to make sure you make use of `categories` and `regressions`.
        Override it with your additional arguments such as `images`.

        Args:
            model_out (Any): Inference output.
            categories (Optional[Sequence[InferCategory]], optional): List of inference categories. Defaults to None.
            regressions (Optional[Sequence[InferRegression]], optional): List of inference regressions. Defaults to None.

        Returns:
            Any: Formatted output.
        """

    @abstractmethod
    def __call__(self, **net_input) -> Any:
        """
        Abstract method to run inference model.

        This method orchestrates the entire inference process: preprocessing,
        inference, postprocessing, and output formatting.

        Returns:
            Any: The formatted inference result.
        """

        x = self.preprocess(**net_input)
        out = self.inference(x)
        out = self.postprocess(out)

        result = self.output_formatter(
            out, categories=self.categories, regressions=self.regressions, **net_input
        )

        return result
