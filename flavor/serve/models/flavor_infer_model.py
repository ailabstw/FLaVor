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


def check_any_nonint(x):
    return np.any(~(np.mod(x, 1) == 0))


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


class InferClassificationOutput(BaseModel, arbitrary_types_allowed=True, protected_namespaces=()):
    images: Sequence[AiImage]
    categories: Sequence[InferCategory]
    model_out: np.ndarray

    @model_validator(mode="before")
    @classmethod
    def check_model_out(cls, data: Any) -> Any:
        model_out = data.get("model_out")

        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim != 1:
            raise ValueError(
                f"The dimension of `model_out` should be in 1D but got {model_out.ndim}."
            )

        if check_any_nonint(model_out):
            raise ValueError(
                "The value of `model_out` should be only 0 or 1 with int or float type."
            )

        return data


class DetModelOut(BaseModel, arbitrary_types_allowed=True):
    bbox_pred: Union[np.ndarray, Sequence[np.ndarray], Sequence[Sequence[int]]]
    cls_pred: Union[np.ndarray, Sequence[np.ndarray], Sequence[Sequence[Union[int, float]]]]
    confidence_score: Optional[Union[np.ndarray, Sequence[float]]] = None
    regression_value: Optional[Union[np.ndarray, Sequence[float]]] = None

    @model_validator(mode="before")
    @classmethod
    def check_model_out(cls, data: Any) -> Any:
        bbox_pred = data.get("bbox_pred")
        cls_pred = data.get("cls_pred")
        confidence_score = data.get("confidence_score", None)
        regression_value = data.get("regression_value", None)

        if len(bbox_pred) != len(cls_pred):
            raise ValueError("`bbox_pred` and `cls_pred` should have same amount of elements.")

        if confidence_score is not None and len(bbox_pred) != len(confidence_score):
            raise ValueError(
                "`bbox_pred` and `confidence_score` should have same amount of elements."
            )

        if regression_value is not None and len(bbox_pred) != len(regression_value):
            raise ValueError(
                "`bbox_pred` and `regression_value` should have same amount of elements."
            )

        if not isinstance(cls_pred, np.ndarray) and not isinstance(cls_pred, list):
            raise TypeError(
                f"`cls_pred` must be type: np.ndarray or list but got {type(cls_pred)}."
            )

        if check_any_nonint(cls_pred):
            raise ValueError(
                "The value of `cls_pred` should be only 0 or 1 with int or float type."
            )

        return data


class InferDetectionOutput(BaseModel, protected_namespaces=()):
    images: Sequence[AiImage]
    categories: Sequence[InferCategory]
    regressions: Optional[Sequence[InferRegression]] = None
    model_out: DetModelOut


class InferRegressionOutput(BaseModel, arbitrary_types_allowed=True, protected_namespaces=()):
    images: Sequence[AiImage]
    regressions: Sequence[InferRegression]
    model_out: np.ndarray

    @model_validator(mode="before")
    @classmethod
    def check_model_out(cls, data: Any) -> Any:
        model_out = data.get("model_out")

        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim != 1 and model_out.ndim != 4:
            raise ValueError(
                f"The dimension of `model_out` should be in 1D but got {model_out.ndim}."
            )

        return data


class InferSegmentationOutput(BaseModel, arbitrary_types_allowed=True, protected_namespaces=()):
    images: Sequence[AiImage]
    categories: Sequence[InferCategory]
    model_out: np.ndarray

    @model_validator(mode="before")
    @classmethod
    def check_model_out(cls, data: Any) -> Any:
        model_out = data.get("model_out")

        if not isinstance(model_out, np.ndarray):
            raise TypeError(f"`model_out` must be type: np.ndarray but got {type(model_out)}.")

        if model_out.ndim != 3 and model_out.ndim != 4:
            raise ValueError(
                f"The dimension of `model_out` should be in 3D or 4D but got {model_out.ndim}."
            )

        if check_any_nonint(model_out):
            raise ValueError(
                "The value of `model_out` should be integer such as 0, 1, 2 ... with int or float type."
            )

        return data


InferOutput = Union[
    InferClassificationOutput,
    InferDetectionOutput,
    InferRegressionOutput,
    InferSegmentationOutput,
]

ModelOut = Union[np.ndarray, DetModelOut]
