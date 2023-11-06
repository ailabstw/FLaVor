from fastapi import FastAPI
from abc import ABC
from ..middlewares import QueueMiddleware


class BaseInvocationAPP(ABC):
    def __init__(self):
        self.app = FastAPI()

        self.app.add_middleware(QueueMiddleware)
