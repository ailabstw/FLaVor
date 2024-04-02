from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def apply(self, body):
        pass
