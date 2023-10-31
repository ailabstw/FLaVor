from typing import Any, Callable, Optional, Type

import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.gzip import GZipMiddleware

from .invocations import InferInvocationAPP


class BaseAPP(object):
    def __init__(self):

        self.app = FastAPI()

        # Add a 'ping' endpoint for health checks or service availability checks
        self.app.add_api_route(
            "/ping",
            self.ping,
            methods=["get"],
        )

    async def ping(self):
        return Response(status_code=204)

    def run(self, host="0.0.0.0", port=9000):
        # Run the FastAPI application using uvicorn
        print(f"listen on port {port}")
        uvicorn.run(self.app, host=host, port=port, log_level="error")


class InferAPP(BaseAPP):
    def __init__(
        self,
        infer_function: Callable,
        input_strategy: Optional[Type[Any]] = None,
        output_strategy: Optional[Type[Any]] = None,
    ):

        super().__init__()

        # Mount the InferInvocationAPP which handles model inference
        self.app.mount(
            path="",
            app=InferInvocationAPP(infer_function, input_strategy, output_strategy).app,
        )

        # Add middleware to compress responses using gzip
        self.app.add_middleware(GZipMiddleware)


class CustomAPP(BaseAPP):
    def __init__(
        self,
        invocation_app: Callable,
    ):

        super().__init__()

        # Mount the invocation_app
        self.app.mount(
            path="",
            app=invocation_app.app,
        )

        self.app.add_middleware(GZipMiddleware)
