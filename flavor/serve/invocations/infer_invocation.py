import json
import logging
import pathlib
import traceback
from collections.abc import Iterable
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable

import aiofile
from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from .base_invocation import BaseInvocationAPP


class InferInvocationAPP(BaseInvocationAPP):
    def __init__(
        self,
        infer_function: Callable,
        input_data_model: BaseModel,
        output_data_model: BaseModel,
    ):

        super().__init__()

        self.app.add_api_route(
            "/invocations",
            self.invocations,
            methods=["post"],
        )

        self.infer_function = infer_function
        self.input_data_model = input_data_model
        self.output_data_model = output_data_model

    async def save_temp_files(self, input_dict, tempdir):
        for k in input_dict:
            if isinstance(input_dict[k], Iterable) and all(
                (isinstance(f, UploadFile) for f in input_dict[k])
            ):
                # save temp file
                filenames = []
                for file_ in input_dict[k]:
                    path = pathlib.Path(file_.filename)
                    suffix = str(path).replace("/", "_")
                    temp_file = NamedTemporaryFile(
                        delete=False, dir=tempdir.name, suffix=f"{suffix}"
                    )
                    async with aiofile.async_open(temp_file.name, "wb") as f:
                        await f.write(await file_.read())
                    filenames.append(temp_file.name)
                input_dict[k] = filenames

        return input_dict

    def deserialize(self, form_data):
        input_dict = {}
        for k in form_data:
            if isinstance(form_data[k], str):
                input_dict[k] = json.loads(form_data[k])
            elif isinstance(form_data[k], UploadFile) or issubclass(form_data[k], UploadFile):
                input_dict[k] = form_data.getlist(k)
            else:
                input_dict[k] = form_data[k]

        return input_dict

    async def invocations(self, request: Request):
        form_data = await request.form()

        try:
            input_dict = self.deserialize(form_data)
            self.input_data_model.model_validate(input_dict)
            tempdir: TemporaryDirectory = TemporaryDirectory()
            input_dict = await self.save_temp_files(input_dict, tempdir)
            response = self.infer_function(**input_dict)
            response = self.output_data_model.model_validate(response)
        except Exception:
            err_msg = traceback.format_exc()
            logging.error(err_msg)
            return JSONResponse(content=err_msg, status_code=status.HTTP_400_BAD_REQUEST)

        tempdir.cleanup()

        return JSONResponse(content=jsonable_encoder(response), status_code=status.HTTP_200_OK)
