import numpy as np
import pytest

from flavor.serve.inference.base_triton_inference_model import (
    TritonInferenceModel,
    TritonInferenceModelSharedSystemMemory,
    BaseTritonClient,
)

client = BaseTritonClient(triton_url="localhost:8000")


@pytest.mark.skipif(
    client.model_configs.get("echo", {}).get("state") != "READY", reason=f"model: echo is not READY"
)
def test_triton_inference_model():
    model = TritonInferenceModelSharedSystemMemory(
        triton_url="localhost:8000",
        model_name="echo",
        model_version=1,
    )

    data_dict = {
        "bool": np.array([False]),
        "uint": np.array([2]),
        "int": np.random.randint(0, 1, (3, 3)),
        "images": np.random.rand(1, 3, 2, 2),
        "string": np.array(["hello", "world"]),
    }
    out = model.forward(data_dict)

    assert np.array_equal(out["int"], data_dict["int"])
    assert np.array_equal(out["string"], data_dict["string"])
    del model


@pytest.mark.skipif(
    client.model_configs.get("resnet50", {}).get("state") != "READY",
    reason=f"model: resnet50 is not READY",
)
def test_triton_shm_inference_model():
    """
    resnet50 is onnx model of a torchvision model converted with torch.
    Loaded to Triton inference server with minimal config.pbtxt:

        dynamic_batching { }
        instance_group [
            {
                count: 1
                kind: KIND_CPU
            }
        ]

        model_warmup  [
        {
            name: "data"
            batch_size: 4
            inputs: {
            key: "data"
            value: {
                data_type: TYPE_FP32
                dims: [3, 224, 224]
                random_data: true
            }
            }
        }
        ]

    with its output name un-specified when converting to onnx, the output name is `resnetv17_dense0_fwd` by default.

    In this test case, client instance is deleted and tests are repeated to ensure shm destructor works properly.
    """
    for _ in range(2):
        model = TritonInferenceModelSharedSystemMemory(
            triton_url="localhost:8000",
            model_name="resnet50",
            model_version=1,
        )

        out = model.forward(
            {
                "data": np.random.rand(1, 3, 224, 224),
            }
        )

        assert out["resnetv17_dense0_fwd"].shape == (1, 1000)

        del model
