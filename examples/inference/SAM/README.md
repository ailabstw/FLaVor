# Triton Inference Server

FLaVor inference service supports inference operation by Triton Inference Server. We provide interface to interact with models deployed on Triton Inference Server. Refer to [`TritonInferenceModel`](flavor/serve/inference/base_triton_inference_model.py) for more detail

## Segment Anything Model (SAM) Triton Inference Server Integration

The SAM model is a computer vision model designed for segmenting objects in images. It consists of two main components: an encoder and a decoder. The encoder generates image embeddings, while the decoder uses these embeddings and user-provided prompt to segment the desired objects.

This integration allows you to interact with SAM deployed on a Triton Inference Server, enabling efficient and scalable inference for the encoder and decoder components.

### Code Structure

The example includes the following Python files:

1. `sam_triton_inference_model.py`: This file contains two classes:

   * `SamEncoderTritonInferenceModel`: Handles the forward operation for the SAM encoder model on the Triton Inference Server.
   * `SamDecoderTritonInferenceModel`: Handles the forward operation for the SAM decoder model on the Triton Inference Server.

2. `sam_encoder.py`: This file defines the `SamEncoderAiCOCOInferenceModel` class under FLaVor inference service, which is responsible for the SAM encoder inference process. It utilizes the `SamEncoderTritonInferenceModel` class from `sam_triton_inference_model`.py to perform the forward operation on the Triton Inference Server.
3. `sam_decoder.py`: This file defines the `SamAiCOCODecoderInferenceModel` class under FLaVor inference service, which is responsible for the SAM decoder inference process. It utilizes the `SamDecoderTritonInferenceModel` class from `sam_triton_inference_model.py` to perform the forward operation on the Triton Inference Server.
