import tempfile
from typing import Any, Callable, Optional, Type

import aiofile
import starlette
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from .base_invocation import BaseInvocationAPP


class InferInvocationAPP(BaseInvocationAPP):
    def __init__(
        self,
        infer_function: Callable,
        input_strategy: Optional[Type[Any]] = None,
        output_strategy: Optional[Type[Any]] = None,
    ):

        super().__init__()

        self.app.add_api_route(
            "/invocations",
            self.invocations,
            methods=["post"],
        )

        self.infer_function = infer_function
        self.input_strategy = input_strategy() if input_strategy else None
        self.output_strategy = output_strategy() if output_strategy else None

    async def invocations(self, request: Request):

        form_data = await request.form()

        with tempfile.TemporaryDirectory() as tempdir:

            data = {}

            for k in form_data:

                if isinstance(form_data[k], starlette.datastructures.UploadFile):

                    filenames = []
                    for file_ in form_data.getlist(k):
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, dir=tempdir, suffix=f"{file_.filename}"
                        )
                        async with aiofile.async_open(temp_file.name, "wb") as f:
                            await f.write(await file_.read())
                        filenames.append(temp_file.name)
                    data[k] = filenames
                else:
                    data[k] = form_data[k]

            try:
                if self.input_strategy:
                    kwargs = await self.input_strategy.apply(data)
                else:
                    kwargs = data

                result = self.infer_function(**kwargs)

                if self.output_strategy:
                    response = await self.output_strategy.apply(**result)
                else:
                    response = result

            except Exception as e:
                response = {"error": str(e)}

        return JSONResponse(content=jsonable_encoder(response))
