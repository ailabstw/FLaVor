from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError
