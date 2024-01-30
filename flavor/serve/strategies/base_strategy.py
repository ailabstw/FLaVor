from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStrategy(ABC):
    @abstractmethod
    def apply(self, form_data: Dict[str, Any]):
        pass
