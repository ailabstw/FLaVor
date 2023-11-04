from fastapi import FastAPI
from abc import ABC
from ..middlewares import QueueMiddleware, TransformFileToFilenameMiddleware


class BaseInvocationAPP(ABC):
    def __init__(self):
        self.app = FastAPI()

        self.app.add_middleware(QueueMiddleware)
        self.app.add_middleware(TransformFileToFilenameMiddleware)
