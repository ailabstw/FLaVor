import ast
import json
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
from fastapi import UploadFile
from pydantic import BaseModel, model_serializer, model_validator

from ..functional import (
    AiAnnotation,
    AiCategory,
    AiHybridMeta,
    AiImage,
    AiMeta,
    AiObject,
    AiRegression,
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


class AiCOCOImageInputDataModel(BaseModel):
    """
    Base class for defining input data model with AiCOCO format.

    Note that you must specify both `images` and `files`. Inherit it if you need extra fields.

    Attributes:
        images (Sequence[AiImage]): Sequence of AiImage objects. Defaults to None.
        files (Sequence[UploadFile]): Sequence of UploadFile objects. Defaults to None.

    Example inheritance:
    ```
    class InputDataModel(AiCOCOImageInputDataModel):
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


class AiCOCOImageOutputDataModel(BaseModel, extra="forbid"):
    """
    Base class for defining image output data model with AiCOCO format.

    Attributes:
        images (Sequence[AiImage]): Sequence of AiImage objects. Defaults to None.
        annotations (Sequence[AiAnnotation]): Sequence of AiAnnotation objects. Defaults to None.
        categories (Sequence[AiCategory]): Sequence of AiCategory objects. Defaults to None.
        regressions (Sequence[AiRegression]): Sequence of AiRegression objects. Defaults to None.
        objects (Sequence[AiObject]): Sequence of AiObject objects. Defaults to None.
        meta (AiMeta): AiMeta object. Defaults to None.

    """

    images: Sequence[AiImage]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiMeta


class AiCOCOTabularInputDataModel(BaseModel):
    """
    Base class for defining tabular input data with AiCOCO format.

    Attributes:
        tables (Sequence[AiTable]): Sequence of AiTable objects.
        meta (AiTableMeta): Metadata with information like window_size.
        files (Sequence[UploadFile]): Sequence of UploadFile objects.

    """

    tables: Sequence[AiTable]
    meta: AiTableMeta
    files: Sequence[UploadFile]


class AiCOCOTabularRecordsArtifact(BaseModel, extra="forbid"):
    """Downloadable artifact reference for tabular AiCOCO records stored as JSON Lines."""

    format: Literal["jsonl"]
    href: str
    rows: int
    bytes: int
    expires_at: Optional[datetime] = None


class AiCOCOTabularOutputDataModel(BaseModel, extra="forbid"):
    """
    Base class for defining tabular output data model with AiCOCO format.

    Tabular inference can produce one record per input row, so records are
    represented as a downloadable JSONL artifact instead of an in-memory response list.

    Attributes:
        tables (Sequence[AiTable]): Processed or transformed tables.
        categories (Sequence[AiCategory]): Categorization results from the data processing.
        regressions (Sequence[AiRegression]): Regression analysis results.
        records (AiCOCOTabularRecordsArtifact): JSONL artifact containing per-row records.
        meta (AiTableMeta): Metadata associated with the tabular output.

    """

    tables: Sequence[AiTable]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    records: AiCOCOTabularRecordsArtifact
    meta: AiTableMeta


class AiCOCOHybridInputDataModel(BaseModel):
    """
    Base class for hybrid input data with AiCOCO format.

    Attributes:
        images (Sequence[AiImage]): Sequence of AiImage objects.
        tables (Sequence[AiTable]): Sequence of AiTable objects.
        meta (AiMeta): Metadata with information like category_ids / regressions / table_ids.
        files (Sequence[UploadFile]): Sequence of UploadFile objects.

    """

    images: Sequence[AiImage]
    tables: Sequence[AiTable]
    meta: AiMeta
    files: Sequence[UploadFile]


class AiCOCOHybridOutputDataModel(BaseModel, extra="forbid"):
    """
    Base class for hybrid output data model with AiCOCO format.

    This class represents the structured output for hybrid data processing,
    combining results from both image and tabular data analysis.

    Attributes:
        images (Sequence[AiImage]): Processed or analyzed image data.
        tables (Sequence[AiTable]): Processed or transformed tabular data.
        annotations (Sequence[AiAnnotation]): Annotations generated during
            the data processing.
        categories (Sequence[AiCategory]): Categorization results.
        regressions (Sequence[AiRegression]): Regression analysis results.
        objects (Sequence[AiObject]): Detected or processed objects.
        meta (AiHybridMeta): Metadata specific to hybrid data processing.

    """

    images: Sequence[AiImage]
    tables: Sequence[AiTable]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiHybridMeta
