"""
This module defines a set of Pydantic models for handling AICOCO format.
"""
from typing import List, Literal, Optional, Sequence

from pydantic import BaseModel


class AiAnnotation(BaseModel, extra="allow"):
    id: str
    image_id: str
    iscrowd: Literal[0, 1]
    object_id: str
    bbox: Optional[Sequence[Sequence[int]]]
    segmentation: Optional[List[Sequence[int]]]


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
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]


class AiMeta(BaseModel, extra="allow"):
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]


class AiImage(BaseModel, extra="allow"):
    file_name: str
    id: str
    index: int
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]


class AiCOCOImageFormat(BaseModel, extra="forbid"):
    images: Sequence[AiImage]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiMeta


class AiTable(BaseModel, extra="allow"):
    id: str
    file_name: str


class AiInstance(BaseModel, extra="allow"):
    id: str
    table_id: str
    row_indexes: Sequence[int]
    category_ids: Optional[Sequence[str]]
    regressions: Optional[Sequence[AiRegressionItem]]


class AiTableMeta(BaseModel, extra="allow"):
    window_size: int


class AiCOCOTabularFormat(BaseModel, extra="forbid"):
    tables: Sequence[AiTable]
    categories: Optional[Sequence[AiCategory]]
    regressions: Optional[Sequence[AiRegression]]
    instances: Sequence[AiInstance]
    meta: AiTableMeta
