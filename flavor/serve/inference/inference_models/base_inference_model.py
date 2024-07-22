from abc import ABC, abstractmethod


class BaseInferenceModel(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
