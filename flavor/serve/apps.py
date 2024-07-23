import traceback
from typing import Callable, Optional, Sequence, Type

import gradio as gr
import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from .invocations import InferInvocationAPP

__all__ = ["InferAPP", "GradioInferAPP"]


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
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def run(self, host="0.0.0.0", port=9000, log_level="error"):
        # Run the FastAPI application using uvicorn
        print(f"listen on port {port}")
        uvicorn.run(self.app, host=host, port=port, log_level=log_level)


class InferAPP(BaseAPP):
    def __init__(
        self,
        infer_function: Callable,
        input_data_model: Optional[Type[BaseModel]] = None,
        output_data_model: Optional[Type[BaseModel]] = None,
    ):

        super().__init__()

        # Mount the InferInvocationAPP which handles model inference
        self.app.mount(
            path="",
            app=InferInvocationAPP(infer_function, input_data_model, output_data_model).app,
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


class GradioInferAPP:
    def __init__(
        self,
        infer_function: Callable,
    ):
        self.infer_function = infer_function

    def invocations(self, files: Sequence[str]):
        try:
            response = self.infer_function(files=files)
        except Exception:
            err_msg = traceback.format_exc()
            return None, None, None, err_msg

        return response

    def run(self, port=9111, **kwargs):

        iface = gr.Interface(
            fn=self.invocations,
            inputs=gr.File(label="DICOM files", file_count="multiple"),
            outputs=[
                gr.Gallery(label="Image"),
                gr.Gallery(label="Prediction"),
                gr.Label(label="Label"),
                gr.Textbox(label="Status"),
            ],
        )
        iface.launch(server_port=port)
