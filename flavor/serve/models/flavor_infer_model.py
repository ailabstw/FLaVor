from typing import Dict, Optional, Sequence, Union

import numpy as np
from pydantic import AfterValidator, BaseModel
from typing_extensions import Annotated

from . import AiImage, AiRegressionItem


class InferInputImage(BaseModel):
    category_ids: Optional[Sequence[str]]
    file_name: str
    id: str
    index: int
    regressions: Optional[Sequence[AiRegressionItem]]
    physical_file_name: str


class InferCategories(BaseModel):
    name: str
    supercategory_name: Optional[str] = None
    display: Optional[bool] = True
    color: Optional[str] = None


class InferRegressions(BaseModel):
    name: str
    superregression_name: Optional[str] = None
    unit: Optional[str] = None


class _ModelOut(BaseModel, arbitrary_types_allowed=True, protected_namespaces=()):
    model_out: np.ndarray


class _ClsPred(BaseModel, arbitrary_types_allowed=True):
    cls_pred: Union[np.ndarray, Sequence[np.ndarray], Sequence[int]]


class InferDetectionModelOutput(BaseModel):
    bbox_pred: Sequence[Sequence[int]]
    _ClsPred
    confidence_score: Optional[float] = None
    regression_value: Optional[float] = None


def check_cls_dim(v):
    assert v.ndim == 1, f"dim of the inference model output {v.shape} should be 1D."
    return v


def check_det_dim(v):
    assert isinstance(v, dict), "The type of inference model output should be `dict`."
    assert "bbox_pred" in v, "A key `bbox_pred` must be in inference model output."
    assert "cls_pred" in v, "A key `cls_pred` must be in inference model output."
    return v


def check_reg_dim(v):
    assert v.ndim == 1, f"dim of the inference model output {v.shape} should be 1D."
    return v


def check_seg_dim(v):
    assert (
        v.ndim == 3 or v.ndim == 4
    ), f"dim of the inference model output {v.shape} should be in 3D or 4D."
    return v


ClsModelOut = Annotated[_ModelOut, AfterValidator(check_cls_dim)]
RegModelOut = Annotated[_ModelOut, AfterValidator(check_reg_dim)]
DetModelOut = Annotated[InferDetectionModelOutput, AfterValidator(check_det_dim)]
SegModelOut = Annotated[_ModelOut, AfterValidator(check_seg_dim)]


class InferClassificationOutput(BaseModel):
    sorted_images: Sequence[AiImage]
    categories: Dict[int, InferCategories]
    ClsModelOut


class InferDetectionOutput(BaseModel):
    sorted_images: Sequence[AiImage]
    categories: Dict[int, InferCategories]
    regressions: Optional[Dict[int, InferRegressions]]
    DetModelOut


class InferRegressionOutput(BaseModel):
    sorted_images: Sequence[AiImage]
    regressions: Dict[int, InferRegressions]
    RegModelOut


class InferSegmentationOutput(BaseModel):
    sorted_images: Sequence[AiImage]
    categories: Dict[int, InferCategories]
    SegModelOut


InferOutput = Union[
    InferClassificationOutput,
    InferDetectionModelOutput,
    InferRegressionOutput,
    InferSegmentationOutput,
]
ModelOutput = Union[ClsModelOut, RegModelOut, DetModelOut, SegModelOut]
