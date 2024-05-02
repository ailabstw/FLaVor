from abc import ABC, abstractclassmethod


class BaseStrategy(ABC):
    @abstractclassmethod
    def __call__(self):
        pass
