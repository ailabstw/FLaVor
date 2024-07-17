"""
This module defines a set of Pydantic models for handling AICOCO format.
"""
from typing import Literal, Optional, Sequence

from pydantic import BaseModel


class AiAnnotation(BaseModel, extra="allow"):
    id: str
    image_id: str
    iscrowd: Literal[0, 1]
    object_id: str
    bbox: Optional[Sequence[Sequence[int]]]
    segmentation: Optional[Sequence[Sequence[int]]]


class AiCategory(BaseModel, extra="allow"):
    id: str
    name: str
    supercategory_id: Optional[str]


class AiRegression(BaseModel, extra="allow"):
    id: str
    name: str
    superregression_id: Optional[str]


class AiRegressionItem(BaseModel, extra="forbid"):
    regression_id: str
    value: float


class AiObject(BaseModel, extra="allow"):
    id: str
    category_ids: Optional[Sequence[str]]
    regressions: Optional[Sequence[AiRegressionItem]]


class AiMeta(BaseModel, extra="allow"):
    category_ids: Optional[Sequence[str]]
    regressions: Optional[Sequence[AiRegressionItem]]


class AiImage(BaseModel, extra="allow"):
    file_name: str
    id: str
    index: int
    category_ids: Optional[Sequence[str]]
    regressions: Optional[Sequence[AiRegressionItem]]


class AiCOCOImageFormat(BaseModel, extra="forbid"):
    images: Sequence[AiImage]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiMeta
