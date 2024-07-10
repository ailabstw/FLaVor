import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_seg_triton():
    seg_triton_example = pytest.importorskip("test_tasks.seg_triton_example")
    seg_triton_app = seg_triton_example.app

    files = []
    filepath = "examples/inference/test_data/seg/300.png"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/seg/input_seg.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(
        transport=ASGITransport(app=seg_triton_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200
