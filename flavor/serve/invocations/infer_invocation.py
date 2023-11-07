from typing import Callable, Optional

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
        input_strategy: Optional[BaseStrategy] = None,
        output_strategy: Optional[BaseStrategy] = None,
    ):

        super().__init__()

        self.app.add_api_route(
            "/invocations",
            self.invocations,
            methods=["post"],
        )
        self.app.add_middleware(TransformFileToFilenameMiddleware)

        self.infer_function = infer_function
        self.input_strategy = input_strategy
        self.output_strategy = output_strategy

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
            response = {"error": str(e)}

        return JSONResponse(content=jsonable_encoder(response))
