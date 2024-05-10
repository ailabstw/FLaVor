import glob
import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from examples.inference.seg3d_example import app

files = []
for filepath in glob.glob("examples/inference/test_data/seg/img0062/*"):
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

with open("examples/inference/test_data/seg/input_3d_dcm.json", "r") as f:
    data = json.load(f)

for k in data:
    data[k] = json.dumps(data[k])


@pytest.mark.asyncio
async def test_seg3d():
    async with AsyncClient(app=app.app, base_url="http://test") as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200
