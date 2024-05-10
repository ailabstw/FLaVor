"""
use `httpx.AsyncClient` to test asynchronous functions.

A minimal example looks like this:

    import pytest
    from httpx import AsyncClient
    from sam_encoder_app import sam_encoder_app

    @pytest.mark.asyncio
    async def test_ping_encoder():
        async with AsyncClient(app=sam_encoder_app.app, base_url="http://") as client:
            response = await client.get("/ping")

A few things to point out:

1. `@pytest.mark.asyncio` is essential.
2. `AsyncClient(..., base_url="http://")` is also essential. You can put whatever after `http://` in base_url.
"""

import json

import numpy as np
import pytest
from httpx import AsyncClient

from examples.inference.SAM.sam_decoder import sam_decoder_app
from examples.inference.SAM.sam_encoder import sam_encoder_app
from examples.inference.SAM.sam_triton_inference_model import (
    SamDecoderInferenceModel,
    SamEncoderInferenceModel,
)
from flavor.serve.models.flavor_infer_model import NpArray


@pytest.mark.asyncio
async def test_ping_encoder():
    async with AsyncClient(app=sam_encoder_app.app, base_url="http://") as client:
        response = await client.get("/ping")
        assert response.status_code == 204


@pytest.mark.asyncio
async def test_ping_decoder():
    async with AsyncClient(app=sam_decoder_app.app, base_url="http://") as client:
        response = await client.get("/ping")
        assert response.status_code == 204


@pytest.mark.asyncio
async def test_encoder():
    filepath = "examples/hello-inference/test_data/SAM/cat.jpg"

    file = open(filepath, "rb")
    files = [("files", (filepath, file))]

    payload = {
        "images": [
            {
                "id": "0",
                "index": 0,
                "file_name": filepath.split("/")[-1],
                "category_ids": None,
                "regressions": None,
            }
        ]
    }
    for k in payload:
        payload[k] = json.dumps(payload[k])

    async with AsyncClient(app=sam_encoder_app.app, base_url="http://") as client:
        res = await client.post("/invocations", data=payload, files=files)

    data = res.json()
    embeddings = NpArray(**data["embeddings"])
    assert embeddings.shape == (1, 256, 64, 64)
    assert len(data["original_shapes"]) == 2

    triton_sam_encoder = SamEncoderInferenceModel(triton_url="triton.user-hannchyun-chen:8000")
    res = triton_sam_encoder.predict([filepath])
    assert np.array_equal(embeddings.array, res["embeddings"])


@pytest.mark.asyncio
async def test_decoder():
    filepath = "examples/hello-inference/test_data/SAM/cat.jpg"

    images = [
        {
            "id": "0",
            "index": 0,
            "file_name": filepath.split("/")[-1],
            "category_ids": None,
            "regressions": None,
        }
    ]

    arr = NpArray(array=np.random.rand(1, 256, 64, 64).astype(np.float32))
    prev_mask = NpArray(array=np.random.rand(1, 1, 256, 256).astype(np.float32))

    fake_input = {
        "image_embeddings": arr.array,
        "point_coords": np.array([[[500, 500]]]),
        "point_labels": np.array([[1]]),
        "orig_im_size": np.array([1500, 1435]),
    }

    triton_sam_decoder = SamDecoderInferenceModel(triton_url="triton.user-hannchyun-chen:8000")
    res = triton_sam_decoder.predict(fake_input)
    mask_bin = (res["masks"] > 0).astype(np.uint8)

    fake_input = {
        "image_embeddings": arr.model_dump(),
        "point_coords": [[[500, 500]]],
        "orig_im_size": [1500, 1435],
        "images": images,
    }

    payload = {k: json.dumps(v) for k, v in fake_input.items()}

    async with AsyncClient(app=sam_decoder_app.app, base_url="http://") as client:
        res = await client.post("/invocations", data=payload)
        assert res.status_code == 200
        ret_wo_prev_mask = res.json()
        assert ret_wo_prev_mask["images"] == images
        assert ret_wo_prev_mask["regressions"] == []
        assert len(ret_wo_prev_mask["annotations"]) == 1

    assert np.array_equal(mask_bin, NpArray(**ret_wo_prev_mask["mask_bin"]).array)

    fake_input = {
        "image_embeddings": arr.model_dump(),
        "point_coords": [[[500, 500]]],
        "orig_im_size": [1500, 1435],
        "images": images,
        "mask_input": prev_mask.model_dump(),
    }

    payload = {k: json.dumps(v) for k, v in fake_input.items()}

    async with AsyncClient(app=sam_decoder_app.app, base_url="http://") as client:
        res = await client.post("/invocations", data=payload)
        assert res.status_code == 200
        ret_with_prev_mask = res.json()
        assert ret_with_prev_mask["images"] == images
        assert ret_with_prev_mask["regressions"] == []
        assert len(ret_with_prev_mask["annotations"]) == 1

    assert not np.array_equal(ret_with_prev_mask["mask_logits"], ret_wo_prev_mask["mask_logits"])


@pytest.mark.asyncio
async def test_integration_flavor():
    filepath = "examples/hello-inference/test_data/SAM/shapes.png"
    # filepath = "examples/hello-inference/test_data/SAM/0.dcm"

    file = open(filepath, "rb")
    files = [("files", (filepath, file))]

    images = [
        {
            "id": "0",
            "index": 0,
            "file_name": filepath.split("/")[-1],
            "category_ids": None,
            "regressions": None,
        }
    ]

    payload = {
        "images": images,
    }
    for k in payload:
        payload[k] = json.dumps(payload[k])

    async with AsyncClient(app=sam_encoder_app.app, base_url="http://") as client:
        res = await client.post("/invocations", data=payload, files=files)

    data = res.json()

    fake_input = {
        "image_embeddings": data["embeddings"],
        "point_coords": [[[250, 250]]],
        "orig_im_size": data["original_shapes"],
        "images": images,
    }

    payload = {k: json.dumps(v, default=str) for k, v in fake_input.items()}

    async with AsyncClient(app=sam_decoder_app.app, base_url="http://") as client:
        res = await client.post("/invocations", data=payload)

        ret = res.json()
        assert res.status_code == 200
        assert ret["images"] == images
        assert ret["regressions"] == []
        assert len(ret["annotations"]) == 1
        assert "mask_logits" in ret
        assert NpArray(**ret["mask_logits"])
