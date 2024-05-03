import os
from typing import List

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
from dicom2jpg.utils import _pixel_process

from flavor.serve.inference import (
    TritonInferenceModel,
    TritonInferenceModelSharedSystemMemory,
)


class SamEncoderInferenceModel:
    def __init__(
        self,
        triton_url: str,
        model_name: str = "sam_encoder",
        model_version: str = "",
        shape: tuple = (1024, 1024),
        is_shared_memory: bool = False,
    ):
        self.shape = shape
        if is_shared_memory:
            self.triton_client = TritonInferenceModelSharedSystemMemory(
                triton_url, model_name, model_version
            )
        else:
            self.triton_client = TritonInferenceModel(triton_url, model_name, model_version)

    def load_image(self, image_path: str) -> np.ndarray:

        _, file_extension = os.path.splitext(image_path)

        if file_extension == ".dcm":

            dcm = pydicom.dcmread(image_path, force=True)
            color_space = pydicom.dcmread(image_path, force=True).get(
                "PhotometricInterpretation", None
            )

            if color_space in [
                "RGB",
                "YBR_RCT",
                "YBR_ICT",
                "YBR_PARTIAL_420",
                "YBR_FULL_422",
                "YBR_FULL",
                "PALETTECOLOR",
            ]:
                dicom_reader = sitk.ImageFileReader()
                dicom_reader.SetFileName(image_path)
                dicom_reader.ReadImageInformation()

                img = sitk.GetArrayFromImage(dicom_reader.Execute())

                if img.ndim > 3:
                    img = np.squeeze(img, axis=0)
            else:
                img = dcm.pixel_array.astype(float)
                img = _pixel_process(dcm, img)
                if (img.ndim == 2) or (img.ndim == 3 and img.shape[-1] == 1):
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _square_pad(self, ndarray):
        h, w = ndarray.shape[:2]

        maxx = max(h, w)
        pad_x = maxx - w
        pad_y = maxx - h

        out = cv2.copyMakeBorder(
            ndarray,
            0,
            pad_y,
            0,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        return out

    def preprocess(self, filenames):
        def filenames_to_batch(filenames):
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([[58.395, 57.12, 57.375]])

            images = []
            shapes = []
            for filename in filenames:
                img = self.load_image(filename)
                shapes.append(img.shape[:2])

                img = (img - mean) / std
                img = self._square_pad(img)
                img = cv2.resize(img, self.shape)
                img = np.transpose(img, (2, 0, 1))  # hwc -> chw
                img = img.astype(np.float32)

                images.append(img)
            return np.array(images), np.array(shapes)

        batch, shapes = filenames_to_batch(filenames)
        data_dict = {
            "images": batch,
            "original_shapes": shapes,
        }
        return data_dict

    def predict(self, filenames, output_shapes=None):
        data_dict = self.preprocess(filenames)
        inputs = {"images": data_dict["images"]}
        results = self.triton_client.forward(inputs, output_shapes)
        results["original_shapes"] = data_dict["original_shapes"]
        return results


class SamDecoderInferenceModel:
    def __init__(
        self,
        triton_url: str,
        model_name: str = "sam_decoder",
        model_version: str = "",
        encoder_input_shape: tuple = (1024, 1024),
        is_shared_memory: bool = False,
    ):
        """
        Arguments
            orig_im_size:
                (h, w)
        """
        self.encoder_input_shape = encoder_input_shape
        if is_shared_memory:
            self.triton_client = TritonInferenceModelSharedSystemMemory(
                triton_url, model_name, model_version
            )
        else:
            self.triton_client = TritonInferenceModel(triton_url, model_name, model_version)

    def preprocess_points(self, coords: List[list], image_shape: tuple, encoder_input_shape: tuple):
        h0, w0 = image_shape
        h1, w1 = encoder_input_shape

        maxx = max(h0, w0)

        resize_ratio = max(h1, w1) / maxx

        for i, (x, y) in enumerate(coords):
            assert 0 <= x <= w0, f"coordinates should be in range. got {x}"
            assert 0 <= y <= h0, f"coordinates should be in range. got {y}"
            _x = x * resize_ratio
            _y = y * resize_ratio
            coords[i] = [_x, _y]
        return coords

    def preprocess(self, data_dict):
        batch_size = data_dict["point_coords"].shape[0]
        data_dict["point_coords"] = data_dict["point_coords"].astype(np.float32)
        for b in range(batch_size):
            data_dict["point_coords"][b] = self.preprocess_points(
                data_dict["point_coords"][b],
                image_shape=data_dict["orig_im_size"],
                encoder_input_shape=self.encoder_input_shape,
            )

        return data_dict

    def predict(self, data_dict):
        data_dict = self.preprocess(data_dict)
        results = self.triton_client.forward(data_dict)
        return results
