from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Extra


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


class AiRegression(BaseModel):
    id: str
    name: str
    superregression_id: Optional[str]
    unit: Optional[str] = None


class AiRegressionItem(BaseModel):
    class Config:
        extra = Extra.forbid

    regression_id: str
    value: float


class AiObject(BaseModel):
    category_ids: Optional[List[str]]
    confidence: Optional[float] = None
    id: str
    regressions: Optional[List[AiRegressionItem]]


class TaskType(Enum):
    binary = "binary"
    multilabel = "multilabel"
    multiclass = "multiclass"


class AiMeta(BaseModel):
    category_ids: Optional[List[str]]
    regressions: Optional[List[AiRegressionItem]]
    task_type: Optional[TaskType] = None


class AiImage(BaseModel):
    category_ids: Optional[List[str]]
    file_name: str
    id: str
    index: int
    regressions: Optional[List[AiRegressionItem]]


class AiCOCOFormat(BaseModel):
    class Config:
        extra = Extra.forbid

    images: List[AiImage]
    annotations: List[AiAnnotation]
    categories: List[AiCategory]
    regressions: List[AiRegression]
    objects: List[AiObject]
    meta: AiMeta
