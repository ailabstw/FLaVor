import typing
from collections.abc import Callable
from tempfile import TemporaryDirectory, NamedTemporaryFile

import aiofile
from starlette.datastructures import UploadFile
from fastapi import FastAPI, Request

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send, Message


class _CachedRequest(Request):
    """
        Warning: Modification of this class is not recommended.
        It serves as a temporary workaround for versions of Starlette prior to 0.28.0.
    """

    def __init__(self, scope: Scope, receive: Receive):
        super().__init__(scope, receive)
        self._wrapped_rcv_disconnected = False
        self._wrapped_rcv_consumed = False
        self._wrapped_rc_stream = self.stream()

    async def wrapped_receive(self) -> Message:
        if self._wrapped_rcv_disconnected:
            return {"type": "http.disconnect"}
        if self._wrapped_rcv_consumed:
            if self._is_disconnected:
                self._wrapped_rcv_disconnected = True
                return {"type": "http.disconnect"}
            msg = await self.receive()
            if msg["type"] != "http.disconnect":
                raise RuntimeError(f"Unexpected message received: {msg['type']}")
            return msg

        if getattr(self, "_body", None) is not None:
            self._wrapped_rcv_consumed = True
            return {
                "type": "http.request",
                "body": self._body,
                "more_body": False,
            }
        elif self._stream_consumed:
            self._wrapped_rcv_consumed = True
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }
        else:
            try:
                stream = self.stream()
                chunk = await stream.__anext__()
                self._wrapped_rcv_consumed = self._stream_consumed
                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": not self._stream_consumed,
                }
            except ClientDisconnect:
                self._wrapped_rcv_disconnected = True
                return {"type": "http.disconnect"}


class TransformFileToFilenameMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)

        self._tempdir: TemporaryDirectory = TemporaryDirectory()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        self._cached_request = _CachedRequest(scope, receive)

        await super().__call__(scope, receive, send)

    def cleanup(self):
        self._tempdir.cleanup()
        self._tempdir = TemporaryDirectory()

    async def dispatch(self, request: Request, call_next: Callable):
        form_data = await request.form()
        json_body = {}
        for k in form_data:
            if isinstance(form_data[k], UploadFile):
                filenames = []
                for file_ in form_data.getlist(k):
                    temp_file = NamedTemporaryFile(
                        delete=False, dir=self._tempdir.name, suffix=f"{file_.filename}"
                    )
                    async with aiofile.async_open(temp_file.name, "wb") as f:
                        await f.write(await file_.read())
                    filenames.append(temp_file.name)
                json_body[k] = filenames
            else:
                json_body[k] = form_data[k]

        request.state.transformed_body = json_body
        response = await call_next(request)
        self.cleanup()
        return response
