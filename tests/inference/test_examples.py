import json
from pathlib import Path

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_seg():
    from test_tasks.seg_example import app as seg_app

    files = []
    filepath = "examples/inference/test_data/seg/0.dcm"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/seg/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(app=seg_app.app, base_url="http://test") as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_cls():
    from test_tasks.cls_example import app as cls_app

    files = []
    filepath = "chexpert/demo_img.jpg"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/cls/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(app=cls_app.app, base_url="http://test") as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_reg():
    from test_tasks.reg_example import app as reg_app

    files = []
    filepath = "examples/inference/test_data/reg/test.jpeg"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/reg/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(app=reg_app.app, base_url="http://test") as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_det():
    from test_tasks.det_example import app as det_app

    files = []
    filepath = "examples/inference/test_data/det/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/det/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(app=det_app.app, base_url="http://test") as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200
