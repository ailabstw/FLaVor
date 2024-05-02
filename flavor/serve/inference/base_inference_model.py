from abc import ABC, abstractmethod
from typing import Any


class BaseInferenceModel(ABC):
    @abstractmethod
    def __call__(self) -> Any:
        pass
