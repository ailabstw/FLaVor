import ast
import json
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from fastapi import UploadFile
from pydantic import BaseModel, model_serializer, model_validator

from ..functional import AiCOCOImageFormat, AiImage


class NpArray(BaseModel, arbitrary_types_allowed=True):
    array: np.ndarray
    shape: Tuple[int, ...]
    dtype: str

    @model_validator(mode="before")
    @classmethod
    def set_arr_attrs(cls, data: Any) -> Dict[str, Any]:
        if isinstance(data, np.ndarray):
            out = {
                "array": data,
                "shape": data.shape,
                "dtype": data.dtype.name,
            }
            return out
        else:
            array = data.get("array")
            shape = data.get("shape")
            dtype = data.get("dtype")

            if type(array) == np.ndarray:
                if shape is not None or dtype is not None:
                    raise ValueError(
                        "The shape and dtype should be `None` if array is `np.ndarray`."
                    )

            elif type(array) == str:
                # deserialize
                if shape is None or dtype is None:
                    raise ValueError(
                        "The shape and dtype cannot be `None` if array is string representation of `np.ndarray`."
                    )
                array = ast.literal_eval(array)
                shape = tuple(ast.literal_eval(shape))
                array = np.frombuffer(array, dtype=getattr(np, dtype)).reshape(shape)

            if type(array) != np.ndarray:
                raise TypeError(f"`array` must have type: np.ndarray but got {type(array)}")

            data["array"] = array
            data["shape"] = array.shape
            data["dtype"] = array.dtype.name
            return data

    @model_serializer
    def serialize(self) -> Dict[str, Any]:
        return {
            "array": str(self.array.tobytes()),
            "shape": json.dumps(self.array.shape),
            "dtype": self.dtype,
        }


class BaseAiCOCOImageInputDataModel(BaseModel):
    """
    Base class for defining input data model with AiCOCO format.

    Inherit it if you need extra fields.

    Note that `images` and `files` could not be `None` at the same time.

    Attributes:
        images (Optional[Sequence[AiImage]]): Sequence of AiImage objects. Defaults to None.
        files (Optional[Sequence[UploadFile]]): Sequence of UploadFile objects. Defaults to None.

    Example:
    ```
    class InputDataModel(BaseAiCOCOImageInputDataModel):
        image_embeddings: NpArray

    InputDataModel(
        {
            "images": ...,
            "image_embeddings": ...
        }
    )
    ```
    """

    images: Optional[Sequence[AiImage]] = None
    files: Optional[Sequence[UploadFile]] = None

    @model_validator(mode="before")
    @classmethod
    def check_images_files(cls, data: Any) -> Any:
        images = data.get("images", None)
        files = data.get("files", None)
        assert images or files, "`images` and `files` could not be `None` at the same time."
        return data


class BaseAiCOCOImageOutputDataModel(AiCOCOImageFormat):
    """
    Base class for defining output data model with AiCOCO format.

    Inherit it if you need extra fields.

    Attributes:
        images (Sequence[AiImage]): Sequence of AiImage objects. Defaults to None.
        annotations (Sequence[AiAnnotation]): Sequence of AiAnnotation objects. Defaults to None.
        categories (Sequence[AiCategory]): Sequence of AiCategory objects. Defaults to None.
        regressions (Sequence[AiRegression]): Sequence of AiRegression objects. Defaults to None.
        objects (Sequence[AiObject]): Sequence of AiObject objects. Defaults to None.
        meta (AiMeta): AiMeta object. Defaults to None.

    Example:
    ```
    class OutputDataModel(BaseAiCOCOImageOutputDataModel):
        mask_bin: NpArray

    OutputDataModel(
        {
            "images": ...,
            "annotations": ...,
            "categories": ...,
            "objects": ...,
            "meta": ...,
            "mask_bin": ...
        }
    )
    ```
    """

    pass
