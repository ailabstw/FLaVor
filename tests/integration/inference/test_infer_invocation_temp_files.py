import io
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from flavor.serve.invocations.infer_invocation import InferInvocationAPP


class BoundedReadUploadFile(UploadFile):
    def __init__(self, filename: str, content: bytes):
        super().__init__(file=io.BytesIO(content), filename=filename)
        self.read_sizes = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size == -1:
            raise AssertionError("upload file reads must be bounded")
        return await super().read(size)


@pytest.mark.asyncio
async def test_save_temp_files_copies_upload_files_with_bounded_reads():
    payload = b"hello fp tabular inference input"
    upload = BoundedReadUploadFile("infer.csv", payload)
    app = InferInvocationAPP(lambda **kwargs: kwargs, BaseModel, BaseModel)

    with TemporaryDirectory() as tempdir_name:
        tempdir = type("TempDir", (), {"name": tempdir_name})()

        result = await app.save_temp_files({"files": [upload]}, tempdir)

        saved_file = Path(result["files"][0])
        assert saved_file.read_bytes() == payload

    assert upload.read_sizes
    assert all(size > 0 for size in upload.read_sizes)
