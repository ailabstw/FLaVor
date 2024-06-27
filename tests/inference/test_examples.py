import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_seg():
    seg_example = pytest.importorskip("test_tasks.seg_example")
    seg_app = seg_example.app

    files = []
    filepath = "examples/inference/test_data/seg/0.dcm"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/seg/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(
        transport=ASGITransport(app=seg_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_cls():
    cls_example = pytest.importorskip("test_tasks.cls_example")
    cls_app = cls_example.app

    files = []
    filepath = "examples/inference/test_data/cls_reg/n02123159_tiger_cat.jpeg"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/cls_reg/input.json", "r") as f:
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
    reg_example = pytest.importorskip("test_tasks.reg_example")
    reg_app = reg_example.app

    files = []
    filepath = "examples/inference/test_data/cls_reg/n02123159_tiger_cat.jpeg"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/cls_reg/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(
        transport=ASGITransport(app=reg_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_det():
    det_example = pytest.importorskip("test_tasks.det_example")
    det_app = det_example.app

    files = []
    filepath = "examples/inference/test_data/det/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg"
    filepath = Path(filepath)
    file = open(filepath, "rb")
    files.append(("files", (f"_{filepath.parent.stem}_{filepath.name}", file)))

    with open("examples/inference/test_data/det/input.json", "r") as f:
        data = json.load(f)

    for k in data:
        data[k] = json.dumps(data[k])

    async with AsyncClient(
        transport=ASGITransport(app=det_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=files)
        assert response.status_code == 200
