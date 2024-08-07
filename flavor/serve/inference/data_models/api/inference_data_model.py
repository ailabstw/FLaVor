import ast
import json
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from fastapi import UploadFile
from pydantic import BaseModel, model_serializer, model_validator

from ..functional import (
    AiCOCOImageFormat,
    AiCOCOTabularFormat,
    AiImage,
    AiTable,
    AiTableMeta,
)


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

    Note that you must specify both `images` and `files`. Inherit it if you need extra fields.

    Attributes:
        images (Sequence[AiImage]): Sequence of AiImage objects. Defaults to None.
        files (Sequence[UploadFile]): Sequence of UploadFile objects. Defaults to None.

    Example inheritance:
    ```
    class InputDataModel(BaseAiCOCOImageInputDataModel):
        image_embeddings: NpArray

    InputDataModel(
        {
            "files": ...,
            "images": ...,
            "image_embeddings": ...
        }
    )
    ```
    """

    images: Sequence[AiImage]
    files: Sequence[UploadFile]


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


class BaseAiCOCOTabularInputDataModel(BaseModel):
    """
    Base class for tabular input data with AiCOCO format.

    Attributes:
        tables (Sequence[AiTable]): Sequence of AiTable objects.
        meta (AiTableMeta): Metadata with information like window_size.
        files (Sequence[UploadFile]): Sequence of UploadFile objects.

    """

    tables: Sequence[AiTable]
    meta: AiTableMeta
    files: Sequence[UploadFile]


class BaseAiCOCOTabularOutputDataModel(AiCOCOTabularFormat):
    """
    Base class for tabular output data with AiCOCO tabular format.
    """

    pass
