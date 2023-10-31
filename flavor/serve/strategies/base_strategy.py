from abc import ABC, abstractmethod

from starlette.formparsers import FormData


class BaseStrategy(ABC):
    @abstractmethod
    def apply(self, form_data: FormData):
        pass
