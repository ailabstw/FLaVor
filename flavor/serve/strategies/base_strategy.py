from typing import Any, Dict
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def apply(self, form_data: Dict[str, Any]):
        pass
