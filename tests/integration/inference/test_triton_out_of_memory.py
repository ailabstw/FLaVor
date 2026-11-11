import pytest
from tritonclient.utils import InferenceServerException

from flavor.serve.errors import OutOfMemoryError
from flavor.serve.inference.inference_models import TritonInferenceModel


class FailingClient:
    def __init__(self, err):
        self.err = err

    def infer(self, **kwargs):
        raise self.err


def model_with(err):
    """A TritonInferenceModel whose client raises `err`, without a Triton server."""
    model = object.__new__(TritonInferenceModel)
    model.client = FailingClient(err)
    model.model_name, model.model_version = "m", "1"
    model.set_input_data = lambda data_dict: []
    return model


@pytest.mark.parametrize(
    "message",
    [
        "PyTorch execute failure: CUDA out of memory. Tried to allocate 2.00 GiB",
        "onnx runtime error 6: CUDA failure 2: out of memory",
        "onnx runtime error 6: CUDA error: Failed to allocate memory for requested buffer of size 1024",
        "CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_triton_out_of_memory_becomes_flavor_out_of_memory_error(message):
    err = InferenceServerException(msg=message, status="400")

    with pytest.raises(OutOfMemoryError) as info:
        model_with(err).forward({})

    assert info.value.__cause__ is err


@pytest.mark.parametrize(
    "message",
    [
        "unexpected shape for input 'images'",
        # ONNX Runtime's CPU arena: host memory, not GPU memory
        "onnx runtime error 6: Failed to allocate memory for requested buffer of size 1024",
    ],
)
def test_other_triton_errors_are_raised_unchanged(message):
    err = InferenceServerException(msg=message, status="400")

    with pytest.raises(InferenceServerException) as info:
        model_with(err).forward({})

    assert info.value is err
