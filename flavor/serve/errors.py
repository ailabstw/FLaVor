class OutOfMemoryError(RuntimeError):
    """
    Inference ran out of GPU memory.

    `InferAPP` answers HTTP 500 with the header `X-FLaVor-Error: CUDA_OUT_OF_MEMORY` for this
    error, as it does for `torch.cuda.OutOfMemoryError`. Raise it (or raise another error from it)
    when the out-of-memory error is not a PyTorch one, e.g. one reported by a Triton server or by
    another framework. The Triton inference models raise it themselves.
    """
