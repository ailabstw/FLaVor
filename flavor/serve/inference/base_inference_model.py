import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence

from pydantic import ValidationError

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
                try:
                    for c in self.categories:
                        InferCategory.model_validate(c)
                except ValidationError:
                    logging.error(
                        "Each element of `categories` should have format of `InferCategory`."
                    )
                    raise
            else:
                raise TypeError("`categories` should have type of `Sequence[InferCategory]`.")

    def check_regressions(self):
        """
        Check if defined regressions field is valid for AiCOCO format.
        """
        if self.regressions is not None:
            if isinstance(self.regressions, Sequence):
                try:
                    for r in self.regressions:
                        InferRegression.model_validate(r)
                except ValidationError:
                    logging.error(
                        "Each element of `regressions` should have format of `InferRegression`."
                    )
                    raise
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
            out, images=self.images, categories=self.categories, regressions=self.regressions
        )

        return result
