from asyncio import Future, Queue, ensure_future
from collections.abc import Callable
from json import JSONDecodeError
from typing import Awaitable, Tuple

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware


class QueueMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        self.request_queue: Queue[
            Tuple[Request, Callable[[Request], Awaitable[Response]], Future[Response]]
        ] = Queue()
        ensure_future(self.handle_request())
        super().__init__(app)

    async def handle_request(self):
        while True:
            request, call_next, future = await self.request_queue.get()
            try:
                response = await call_next(request)
                future.set_result(response)
            except ValidationError as e:
                e: ValidationError
                future.set_result(
                    JSONResponse(
                        content=jsonable_encoder({"detail": e.errors()}),
                        status_code=status.HTTP_400_BAD_REQUEST,
                    )
                )
            except JSONDecodeError as e:
                e: JSONDecodeError
                future.set_result(
                    JSONResponse(
                        content=jsonable_encoder({"detail": e}),
                        status_code=status.HTTP_400_BAD_REQUEST,
                    )
                )
            except Exception as e:
                future.set_exception(e)

    async def dispatch(self, request: Request, call_next: Callable):
        future = Future()
        await self.request_queue.put((request, call_next, future))
        return await future
