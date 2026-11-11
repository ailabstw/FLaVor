import sys
import types

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from flavor.serve.invocations.infer_invocation import (
    CUDA_OUT_OF_MEMORY,
    FLAVOR_ERROR_HEADER,
    InferInvocationAPP,
    is_cuda_out_of_memory,
)


class FakeOutOfMemoryError(RuntimeError):
    pass


class AnyData(BaseModel):
    model_config = ConfigDict(extra="allow")


@pytest.fixture
def fake_torch(monkeypatch):
    """A stand-in `torch` with `torch.cuda.OutOfMemoryError`, so the tests need no torch."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=FakeOutOfMemoryError)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def post_invocation(infer_function):
    app = InferInvocationAPP(infer_function, AnyData, AnyData)
    with TestClient(app.app) as client:
        return client.post("/invocations", data={"meta": "{}"})


def raise_(err):
    def infer_function(**kwargs):
        raise err

    return infer_function


def raise_wrapped_oom(**kwargs):
    try:
        raise FakeOutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    except FakeOutOfMemoryError as err:
        raise RuntimeError("inference failed") from err


def test_cuda_out_of_memory_returns_500_with_error_header(fake_torch):
    response = post_invocation(raise_(FakeOutOfMemoryError("CUDA out of memory")))

    assert response.status_code == 500
    assert response.headers[FLAVOR_ERROR_HEADER] == CUDA_OUT_OF_MEMORY == "CUDA_OUT_OF_MEMORY"
    assert FLAVOR_ERROR_HEADER == "X-FLaVor-Error"


def test_cuda_out_of_memory_raised_from_another_error_is_detected(fake_torch):
    response = post_invocation(raise_wrapped_oom)

    assert response.status_code == 500
    assert response.headers[FLAVOR_ERROR_HEADER] == CUDA_OUT_OF_MEMORY


def test_other_errors_still_return_400_without_error_header(fake_torch):
    response = post_invocation(raise_(ValueError("bad input")))

    assert response.status_code == 400
    assert FLAVOR_ERROR_HEADER not in response.headers
    assert "ValueError: bad input" in response.json()


def test_successful_inference_is_unchanged(fake_torch):
    response = post_invocation(lambda **kwargs: {"ok": True})

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_runtime_error_mentioning_memory_is_not_out_of_memory(fake_torch):
    assert not is_cuda_out_of_memory(RuntimeError("CUDA out of memory"))


def test_without_torch_nothing_is_out_of_memory(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    assert not is_cuda_out_of_memory(FakeOutOfMemoryError("CUDA out of memory"))


def test_real_torch_out_of_memory_error_is_detected():
    torch = pytest.importorskip("torch")

    assert is_cuda_out_of_memory(torch.cuda.OutOfMemoryError("CUDA out of memory"))
