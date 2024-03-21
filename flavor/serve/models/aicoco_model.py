from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel


class Iscrowd(Enum):
    number_0 = 0
    number_1 = 1


class AiAnnotation(BaseModel):
    bbox: Optional[Sequence[Sequence[int]]]
    id: str
    image_id: str
    iscrowd: Iscrowd
    object_id: str
    segmentation: Optional[Sequence[Sequence[int]]]


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


class AiRegressionItem(BaseModel, extra="forbid"):
    regression_id: str
    value: float


class AiObject(BaseModel):
    category_ids: Optional[Sequence[str]]
    confidence: Optional[float] = None
    id: str
    regressions: Optional[Sequence[AiRegressionItem]]


class TaskType(Enum):
    binary = "binary"
    multilabel = "multilabel"
    multiclass = "multiclass"


class AiMeta(BaseModel):
    category_ids: Optional[Sequence[str]]
    regressions: Optional[Sequence[AiRegressionItem]]
    task_type: Optional[TaskType] = None


class AiImage(BaseModel):
    category_ids: Optional[Sequence[str]]
    file_name: str
    id: str
    index: int
    regressions: Optional[Sequence[AiRegressionItem]]


class AiCOCOFormat(BaseModel, extra="forbid"):
    images: Sequence[AiImage]
    annotations: Sequence[AiAnnotation]
    categories: Sequence[AiCategory]
    regressions: Sequence[AiRegression]
    objects: Sequence[AiObject]
    meta: AiMeta
