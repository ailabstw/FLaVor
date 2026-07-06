import inspect
import json
import logging
import pathlib
import traceback
from collections.abc import Iterable
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Dict, Optional

import aiofile
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from ..inference.records_artifacts import (
    RECORDS_ARTIFACT_MEDIA_TYPE,
    TabularRecordsArtifactStore,
)

UPLOAD_COPY_CHUNK_SIZE_BYTES = 8 * 1024 * 1024


async def copy_upload_file(upload_file: UploadFile, destination: str) -> None:
    async with aiofile.async_open(destination, "wb") as f:
        while True:
            chunk = await upload_file.read(UPLOAD_COPY_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            await f.write(chunk)


class InferInvocationAPP:
    """
    A FastAPI application for handling inference invocations.
    This application exposes API endpoint:

    - `/invocations`: for processing inference requests.
        The endpoint accepts POST requests with form data, deserializes the input,
        validates it using the provided Pydantic models, performs inference, and returns the results.

    Args:
        infer_function (Callable): The inference function to be called with the input data.
        input_data_model (BaseModel): The Pydantic model for validating the input data.
        output_data_model (BaseModel): The Pydantic model for validating the output data.
    """

    def __init__(
        self,
        infer_function: Callable,
        input_data_model: BaseModel,
        output_data_model: BaseModel,
        records_output_dir: Optional[str] = None,
        records_href_prefix: Optional[str] = None,
    ):
        self.app = FastAPI()
        self.infer_function = infer_function
        self.input_data_model = input_data_model
        self.output_data_model = output_data_model
        self.records_artifact_store: Optional[TabularRecordsArtifactStore] = None
        if self.supports_records_artifacts():
            self.records_artifact_store = TabularRecordsArtifactStore(
                output_dir=records_output_dir, href_prefix=records_href_prefix
            )

        self.app.add_api_route(
            "/invocations",
            self.invocations,
            methods=["post"],
        )
        if self.records_artifact_store is not None:
            self.app.add_api_route(
                f"{self.records_artifact_store.href_prefix}/{{artifact_name}}",
                self.records_artifact,
                methods=["get"],
            )

    def supports_records_artifacts(self) -> bool:
        output_fields = getattr(self.output_data_model, "model_fields", {})
        return "records" in output_fields

    async def records_artifact(self, artifact_name: str) -> FileResponse:
        if self.records_artifact_store is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Records artifact not found.",
            )

        try:
            artifact_path = self.records_artifact_store.resolve_path(artifact_name)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Records artifact not found.",
            ) from err

        if not artifact_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Records artifact not found.",
            )

        return FileResponse(
            artifact_path,
            media_type=RECORDS_ARTIFACT_MEDIA_TYPE,
            filename=artifact_name,
        )

    def add_records_artifact_context(self, input_dict: Dict) -> Dict:
        if self.records_artifact_store is None:
            return input_dict

        try:
            parameters = inspect.signature(self.infer_function).parameters
        except (TypeError, ValueError):
            return input_dict

        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "records_output_dir" in parameters:
            input_dict.setdefault(
                "records_output_dir", str(self.records_artifact_store.output_dir)
            )
        if accepts_kwargs or "records_href_prefix" in parameters:
            input_dict.setdefault(
                "records_href_prefix", self.records_artifact_store.href_prefix
            )

        return input_dict

    async def save_temp_files(self, input_dict: Dict, tempdir: TemporaryDirectory) -> Dict:
        """
        Save uploaded files as temporary files and update the input dictionary with the file paths.

        Args:
            input_dict (Dict): The input data dictionary.
            tempdir (TemporaryDirectory): The temporary directory to save the files.

        Returns:
            Dict: The updated input dictionary with file paths.
        """
        for k in input_dict:
            if isinstance(input_dict[k], Iterable) and all(
                (isinstance(f, UploadFile) for f in input_dict[k])
            ):
                # save temp file
                filenames = []
                for file_ in input_dict[k]:
                    path = pathlib.Path(file_.filename)
                    suffix = str(path).replace(
                        "/", "@@@"
                    )  # consist with `flavor/serve/inference/inference_models/base_aicoco_inference_model.py` in L249
                    with NamedTemporaryFile(
                        delete=False, dir=tempdir.name, suffix=f"_{suffix}"
                    ) as temp_file:
                        temp_file_name = temp_file.name
                    await copy_upload_file(file_, temp_file_name)
                    filenames.append(temp_file_name)
                input_dict[k] = filenames

        return input_dict

    def deserialize(self, form_data: Dict) -> Dict:
        """
        Deserialize the form data into a dictionary.

        Args:
            form_data (Dict): The form data from the request.

        Returns:
            dict: The deserialized input dictionary.
        """
        input_dict = {}
        for k in form_data:
            if isinstance(form_data[k], str):
                input_dict[k] = json.loads(form_data[k])
            elif isinstance(form_data[k], UploadFile) or issubclass(form_data[k], UploadFile):
                input_dict[k] = form_data.getlist(k)
            else:
                input_dict[k] = form_data[k]

        return input_dict

    async def invocations(self, request: Request) -> JSONResponse:
        """
        Handle the /invocation requests.

        Args:
            request (Request): The FastAPI request object.

        Returns:
            JSONResponse: The JSON response with the inference results or error message.
        """
        form_data = await request.form()
        tempdir: TemporaryDirectory = TemporaryDirectory()

        try:
            input_dict = self.deserialize(form_data)
            self.input_data_model.model_validate(input_dict)

            input_dict = await self.save_temp_files(input_dict, tempdir)
            input_dict = self.add_records_artifact_context(input_dict)
            response = self.infer_function(**input_dict)
            response = self.output_data_model.model_validate(response)

        except Exception:
            err_msg = traceback.format_exc()
            logging.error(err_msg)
            return JSONResponse(content=err_msg, status_code=status.HTTP_400_BAD_REQUEST)

        finally:
            tempdir.cleanup()

        return JSONResponse(content=jsonable_encoder(response), status_code=status.HTTP_200_OK)
