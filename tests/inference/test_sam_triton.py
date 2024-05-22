import cv2
import numpy as np

from examples.inference.SAM.sam_triton_inference_model import (
    SamDecoderInferenceModel,
    SamEncoderInferenceModel,
)


def test_integration_triton():
    encoder = SamEncoderInferenceModel(triton_url="triton:8000")
    decoder = SamDecoderInferenceModel(triton_url="triton:8000")
    results = encoder.predict(["examples/hello-inference/test_data/SAM/shapes.png"])

    fake_input = {
        "image_embeddings": results["embeddings"],
        "orig_im_size": np.array([730, 1024], dtype=np.float32),
        "point_coords": np.array([[[834, 188]]], dtype=np.float32),
        "point_labels": np.array([[1]], dtype=np.float32),
    }
    results = decoder.predict(fake_input)
    cv2.imwrite("tmp.png", (results["masks"][0][0] > 0).astype(np.uint8) * 255)


def test_decoder_different_coord_dtype():
    triton_sam_decoder = SamDecoderInferenceModel(triton_url="triton:8000")
    arr = np.random.rand(1, 256, 64, 64).astype(np.float32)

    res1 = triton_sam_decoder.predict(
        {
            "image_embeddings": arr,
            "point_coords": np.array([[[500, 500]]]).astype(np.float32),
            "point_labels": np.array([[1]]).astype(np.float32),
            "orig_im_size": np.array([1500, 1435]).astype(np.float32),
        }
    )
    res2 = triton_sam_decoder.predict(
        {
            "image_embeddings": arr,
            "point_coords": np.array([[[500, 500]]]),
            "point_labels": np.array([[1]]),
            "orig_im_size": np.array([1500, 1435]),
        }
    )

    assert np.array_equal(res1["masks"], res2["masks"])


def test_decoder_shared_memory():
    triton_sam_decoder = SamDecoderInferenceModel(triton_url="triton:8000", is_shared_memory=True)

    res1 = triton_sam_decoder.predict(
        {
            "image_embeddings": np.random.rand(1, 256, 64, 64).astype(np.float32),
            "point_coords": np.array([[[500, 500]]]).astype(np.float32),
            "point_labels": np.array([[1]]).astype(np.float32),
            "orig_im_size": np.array([1500, 1435]).astype(np.float32),
            "has_mask_input": np.zeros((1)).astype(np.float32),
            "mask_input": np.zeros((1, 1, 256, 256)).astype(np.float32),
        },
        output_shapes={
            "low_res_masks": (1, 1, 256, 256),
            "iou_predictions": (1, 1),
            "masks": (1, 1, 1500, 1435),
        },
    )

    assert res1["low_res_masks"].shape == (1, 1, 256, 256)
    assert res1["iou_predictions"].shape == (1, 1)
    assert res1["masks"].shape == (1, 1, 1500, 1435)


def test_decoder_http():
    triton_sam_decoder = SamDecoderInferenceModel(triton_url="triton:8000")

    res = triton_sam_decoder.predict(
        {
            "image_embeddings": np.random.rand(1, 256, 64, 64).astype(np.float32),
            "point_coords": np.array([[[500, 500]]]).astype(np.float32),
            "point_labels": np.array([[1]]).astype(np.float32),
            "orig_im_size": np.array([1500, 1435]).astype(np.float32),
            "has_mask_input": np.zeros((1)).astype(np.float32),
            "mask_input": np.zeros((1, 1, 256, 256)).astype(np.float32),
        }
    )

    assert res["low_res_masks"].shape == (1, 1, 256, 256)
    assert res["iou_predictions"].shape == (1, 1)
    assert res["masks"].shape == (1, 1, 1500, 1435)


def test_encoder_http():
    triton_sam_encoder = SamEncoderInferenceModel(triton_url="triton:8000")

    res = triton_sam_encoder.predict(["examples/hello-inference/test_data/SAM/cat.jpg"])

    assert res["embeddings"].shape == (1, 256, 64, 64)
    assert "original_shapes" in res
