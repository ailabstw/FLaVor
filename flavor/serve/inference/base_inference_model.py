import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence

from pydantic import ValidationError

from flavor.serve.models import InferCategory, InferRegression


class BaseInferenceModel(ABC):
    @abstractmethod
    def __init__(self) -> Any:
        pass

    @abstractmethod
    def __call__(self) -> Any:
        pass


class BaseAiCOCOInferenceModel(BaseInferenceModel):
    def __init__(self):
        self.network = self.define_inference_network()
        self.categories = self.set_categories()
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
        self.regressions = self.set_regressions()
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
