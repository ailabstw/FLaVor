from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Extra, Field


class AiImage(BaseModel):
    category_ids: Optional[List[str]] = None
    file_name: str
    id: str
    index: Optional[int] = None


class Iscrowd(Enum):
    number_0 = 0
    number_1 = 1


class AiAnnotation(BaseModel):
    bbox: Optional[List[List[int]]]
    id: str
    image_id: str
    iscrowd: Iscrowd
    object_id: str
    segmentation: Optional[List[List[int]]]


class AiCategory(BaseModel):
    color: Optional[str] = None
    id: str
    name: str
    supercategory_id: Optional[str]


class AiObject(BaseModel):
    category_ids: List[str]
    centroid: Optional[List[int]] = Field(None, max_items=2, min_items=2)
    confidence: Optional[float] = None
    id: str
    regression_value: Optional[float] = None


class AiMeta(BaseModel):
    category_ids: Optional[List[str]] = None
    task_type: Optional[str] = None


class AiCOCOFormat(BaseModel):
    class Config:
        extra = Extra.forbid

    images: List[AiImage]
    annotations: List[AiAnnotation]
    categories: List[AiCategory]
    objects: List[AiObject]
    meta: AiMeta
