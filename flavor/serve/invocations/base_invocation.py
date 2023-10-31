from fastapi import FastAPI

from ..middlewares import QueueMiddleware


class BaseInvocationAPP(object):
    def __init__(self):

        self.app = FastAPI()

        self.app.add_middleware(QueueMiddleware)
