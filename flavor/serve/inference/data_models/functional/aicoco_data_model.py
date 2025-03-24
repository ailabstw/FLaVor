"""
This module defines a set of Pydantic models for handling AICOCO format.
"""
from typing import List, Literal, Optional, Sequence, TypedDict, Union

from pydantic import BaseModel

# ==============================
# Annotation Related Models
# ==============================


class AiAnnotation(BaseModel, extra="allow"):
    id: str
    image_id: str
    iscrowd: Literal[0, 1]
    object_id: str
    bbox: Optional[Sequence[Sequence[int]]]
    segmentation: Optional[List[Sequence[int]]]


# ==============================
# Category and Regression Definitions
# ==============================


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


# ==============================
# Object and Metadata Models
# ==============================


class AiObject(BaseModel, extra="allow"):
    id: str
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]


class AiMeta(BaseModel, extra="allow"):
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]


class AiTableMeta(BaseModel, extra="allow"):
    window_size: int


class AiHybridMeta(BaseModel, extra="allow"):
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]
    table_ids: Optional[List[str]]


# ==============================
# Image Related Models
# ==============================


class AiImage(BaseModel, extra="allow"):
    file_name: str
    id: str
    index: int
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]


# ==============================
# Tabular Related Models
# ==============================


class AiTable(BaseModel, extra="allow"):
    id: str
    file_name: str


class AiRecord(BaseModel, extra="allow"):
    id: str
    table_id: str
    row_indexes: Sequence[int]
    category_ids: Optional[Sequence[str]]
    regressions: Optional[Sequence[AiRegressionItem]]


# ==============================
# Intermediate Models
# ==============================


class AiCOCORef(BaseModel):
    images: List[AiImage]
    categories: List[AiCategory]
    regressions: List[AiRegression]
    meta: Union[AiMeta, AiTableMeta, AiHybridMeta]


class AiCOCOAnnotObj(TypedDict):
    annotations: List[AiAnnotation]
    objects: List[AiObject]


# ==============================
# Full Format Models
# ==============================


class AiCOCOImageFormat(BaseModel, extra="forbid"):
    images: Sequence[AiImage]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiMeta


class AiCOCOTabularFormat(BaseModel, extra="forbid"):
    tables: Sequence[AiTable]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    records: Sequence[AiRecord]
    meta: AiTableMeta


class AiCOCOHybridFormat(BaseModel, extra="forbid"):
    images: Sequence[AiImage]
    tables: Sequence[AiTable]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiHybridMeta
