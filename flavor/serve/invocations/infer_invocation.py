import logging
import traceback
from typing import Callable, Optional, Type

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ..middlewares import TransformFileToFilenameMiddleware
from ..strategies import BaseStrategy
from .base_invocation import BaseInvocationAPP


class InferInvocationAPP(BaseInvocationAPP):
    def __init__(
        self,
        infer_function: Callable,
        input_strategy: Optional[Type[BaseStrategy]] = None,
        output_strategy: Optional[Type[BaseStrategy]] = None,
    ):

        super().__init__()

        self.app.add_api_route(
            "/invocations",
            self.invocations,
            methods=["post"],
        )

        self.app.add_middleware(TransformFileToFilenameMiddleware)

        self.infer_function = infer_function
        self.input_strategy = input_strategy() if input_strategy else None
        self.output_strategy = output_strategy() if output_strategy else None

    async def invocations(self, request: Request):
        body = request.state.transformed_json
        try:
            if self.input_strategy:
                kwargs = {**await self.input_strategy.apply(body)}
            else:
                kwargs = {**body}

            result = self.infer_function(**kwargs)

            if self.output_strategy:
                response = await self.output_strategy.apply(result)
            else:
                response = result

        except Exception as e:

            err_msg = "".join(
                traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            )

            logging.error(err_msg)

            return JSONResponse(content=err_msg, status_code=500)

        return JSONResponse(content=jsonable_encoder(response), status_code=200)
