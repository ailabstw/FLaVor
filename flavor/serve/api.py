import logging
import tempfile

import aiofile
import starlette
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import processor

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class InferAPIApp:
    def __init__(self, callback):

        self.app = FastAPI()
        self.processor = processor.TaiMedimgProcessor(callback=callback)

        self.app.post("/invocations")(self.invocations)
        self.app.get("/ping")(self.ping)

    async def invocations(self, request: Request):
        """
        Handle invocation requests.
        """
        try:
            form_data = await request.form()

            result = None

            # Save temporary file for tmi-thor reader
            with tempfile.TemporaryDirectory() as tempdir:

                kwargs = {}

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
                        kwargs[k] = filenames

                    else:
                        kwargs[k] = form_data[k]

                # Pass arguments for call function to infer engine through processor
                result = self.processor(**kwargs)

            status_code = 200 if result is not None and "error" not in result else 400
            return JSONResponse(content=result, status_code=status_code)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def ping(self):
        """
        Health check endpoint.
        """
        headers = {"cache-control": "no-cache"}
        if self.processor.ready:
            return JSONResponse(content={"status": "available"}, headers=headers, status_code=200)
        else:
            return JSONResponse(content={"status": "unavailable"}, headers=headers, status_code=503)

    def run(self, host="0.0.0.0", port=9000):
        """
        Run the application on the specified host and port.
        """
        logging.info(f"[Server] listen on port {port}")
        uvicorn.run(self.app, host=host, port=port)
