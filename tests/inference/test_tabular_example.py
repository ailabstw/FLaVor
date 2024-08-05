import json
import shutil
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_cls():
    cls_example = pytest.importorskip("test_tasks.tabular_cls_example")
    cls_app = cls_example.app

    files = []
    filepath = "examples/inference/test_data/tabular/cls/test_cls.csv"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/tabular/cls/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(
        transport=ASGITransport(app=cls_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_reg():
    reg_example = pytest.importorskip("test_tasks.tabular_reg_example")
    reg_app = reg_example.app

    files = []
    filepath = "examples/inference/test_data/tabular/reg/test_reg.csv"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/tabular/reg/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(
        transport=ASGITransport(app=reg_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200

    shutil.rmtree(reg_example.MODEL_PATH, ignore_errors=True)
