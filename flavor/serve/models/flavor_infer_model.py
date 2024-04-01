import ast
import json
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, model_serializer, model_validator

from . import AiImage, AiRegressionItem


class NpArray(BaseModel, arbitrary_types_allowed=True):
    array: np.ndarray
    shape: Tuple[int, ...]
    dtype: str

    @model_validator(mode="before")
    @classmethod
    def set_arr_attrs(cls, data):
        array = data.get("array")
        shape = data.get("shape")
        dtype = data.get("dtype")

        if type(array) == np.ndarray:
            if shape is not None or dtype is not None:
                raise ValueError("shape and dtype should be None if array is an `np.ndarray`")

        elif type(array) == str:
            if shape is None or dtype is None:
                raise ValueError(
                    "shape and dtype cannot be None if array is an `np.ndarray` string representation"
                )
            array = ast.literal_eval(array)
            shape = tuple(ast.literal_eval(shape))
            array = np.frombuffer(array, dtype=getattr(np, dtype)).reshape(shape)

        if type(array) != np.ndarray:
            raise TypeError(f"array must have type: np.ndarray. got {type(array)}")

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


class InputBody(BaseModel):
    files: Sequence[str]
    images: str


class InferInputImage(BaseModel):
    category_ids: Optional[Sequence[str]] = None
    file_name: str
    id: str
    index: int
    regressions: Optional[Sequence[AiRegressionItem]] = None
    physical_file_name: str


class InferInput(BaseModel):
    images: Sequence[InferInputImage]


class InferCategory(BaseModel):
    name: str
    supercategory_name: Optional[str] = None
    display: Optional[bool] = True
    color: Optional[str] = None


class InferRegression(BaseModel):
    name: str
    superregression_name: Optional[str] = None
    unit: Optional[str] = None


class DetModelOutput(BaseModel):
    bbox_pred: Sequence[Sequence[int]]
    cls_pred: Any  # TODO add strong constraint
    confidence_score: Optional[float] = None
    regression_value: Optional[float] = None


class InferClassificationOutput(BaseModel):
    images: Sequence[AiImage]
    categories: Dict[int, InferCategory]
    model_out: Any  # TODO add strong constraint


class InferDetectionOutput(BaseModel):
    images: Sequence[AiImage]
    categories: Dict[int, InferCategory]
    regressions: Optional[Dict[int, InferRegression]] = None
    model_out: DetModelOutput


class InferRegressionOutput(BaseModel):
    images: Sequence[AiImage]
    regressions: Dict[int, InferRegression]
    model_out: Any  # TODO add strong constraint


class InferSegmentationOutput(BaseModel):
    images: Sequence[AiImage]
    categories: Dict[int, InferCategory]
    model_out: Any  # TODO add strong constraint


InferOutput = Union[
    InferClassificationOutput,
    InferDetectionOutput,
    InferRegressionOutput,
    InferSegmentationOutput,
]

ModelOutput = Union[np.ndarray, DetModelOutput]
