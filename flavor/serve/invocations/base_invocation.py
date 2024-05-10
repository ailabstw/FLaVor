from abc import ABC

from fastapi import FastAPI


class BaseInvocationAPP(ABC):
    def __init__(self):
        self.app = FastAPI()
